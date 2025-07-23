import pandas as pd
import json
import re
from datetime import datetime
from typing import List, Union, Dict, Any
from _core.config import config
from _core.models import ReflectTask, RelevanceCheck
from _core.logger import custom_logger
from _core.llm_client import ClientManager
from _core.prompts import (
    CREATE_QUERIES,
    CREATE_QUERIES_ADDITIONAL,
    ANALYZE_DOCUMENT,
    DOCUMENT,
    REFLECT_TASK,
    RESEARCH_WRITER,
    FORMAT_RESULT,
    CHECK_RELEVANCE,
)
from _core.models import SearchQueries
from _core.utils import call_function_in_parallel

llm_client = ClientManager().get_client(provider="openrouter")


def _prepare_json_schema(model_class) -> dict:
    """Prepare JSON schema with additionalProperties disabled."""
    schema = model_class.model_json_schema()
    schema["additionalProperties"] = False
    return schema


def _parse_json_response(response: str) -> Dict[str, Any] | None:
    """
    Parse JSON response, handling markdown code blocks and various formats.

    Args:
        response (str): Raw response string

    Returns:
        Dict[str, Any] | None: Parsed JSON or None if parsing fails
    """
    if not response or not isinstance(response, str):
        return None

    try:
        # First try direct parsing
        return json.loads(response.strip())
    except json.JSONDecodeError:
        pass

    try:
        # Handle markdown code blocks
        json_match = re.search(r"```(?:json)?\s*\n(.*?)\n```", response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1).strip())
    except json.JSONDecodeError:
        pass

    custom_logger.error(f"Failed to parse JSON response: {response[:100]}...")
    return None


def _to_bool(value: Union[str, bool, None]) -> bool | None:
    """Convert various boolean representations to actual booleans."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lower_value = value.lower()
        if lower_value in ("true", "1", "yes"):
            return True
        elif lower_value in ("false", "0", "no"):
            return False
    return None


def create_queries(
    prompt: str,
    max_queries: int = config["app"]["max_queries"],
    model_id: str = config["models"]["performance_low"],
    previous_queries: list[str] = [],
    previous_considerations: list[str] = [],
    first_iteration: bool = True,
) -> List[str]:
    """Generate search queries from a prompt and optional context."""
    json_schema = _prepare_json_schema(SearchQueries)
    system_message = CREATE_QUERIES.format(query_count=max_queries)

    if not first_iteration:
        system_message += CREATE_QUERIES_ADDITIONAL.format(
            previous_queries="\n".join(previous_queries),
            considerations="\n".join(previous_considerations),
        )

    response = llm_client.call_structured(
        prompt,
        json_schema,
        model_id=model_id,
        temperature=config["temperature"]["high"],
        system_message=system_message,
    )

    if not response:
        return []

    parsed = _parse_json_response(response)
    return parsed.get("queries", []) if parsed else []


def analyze_documents(
    user_query: str,
    document_ids: List[int],
    data: pd.DataFrame,
    model_id: str = config["models"]["performance_low"],
) -> List[str]:
    """Analyze documents based on a user query using a language model."""
    relevant_docs = data[data["identifier"].isin(document_ids)]

    prompts = [
        ANALYZE_DOCUMENT.format(
            user_query=user_query,
            title=row["title"],
            text=row["text"],
            date=row["date"],
            link=row["link"],
        )
        for _, row in relevant_docs.iterrows()
    ]

    results = call_function_in_parallel(
        prompts,
        llm_client.call,
        max_workers=config["parallelization"]["max_workers"],
        model_id=model_id,
        temperature=config["temperature"]["low"],
    )

    return results or []


def _parse_relevance_results(
    results: List[str],
) -> List[tuple[bool | None, str | None]]:
    """Parse relevance check results into (relevance, reasoning) tuples."""
    checks = []

    for result in results:
        if not result:
            checks.append((None, None))
            continue

        parsed = _parse_json_response(result)
        if parsed:
            relevance = _to_bool(parsed.get("relevance"))
            reasoning = parsed.get("reasoning")
            checks.append((relevance, reasoning))
        else:
            checks.append((None, None))

    return checks


def check_relevance(
    user_query: str,
    data: pd.DataFrame,
    model_id: str = config["models"]["performance_low"],
) -> pd.DataFrame:
    """Check document relevance for a given prompt."""
    prompts = [
        FORMAT_RESULT.format(
            user_query=user_query,
            chunk_text=row["chunk_text"],
        )
        for _, row in data.iterrows()
    ]

    json_schema = _prepare_json_schema(RelevanceCheck)
    results = call_function_in_parallel(
        prompts,
        llm_client.call_structured,
        json_schema=json_schema,
        max_workers=config["parallelization"]["max_workers"],
        model_id=model_id,
        temperature=config["temperature"]["low"],
        system_message=CHECK_RELEVANCE,
    )

    checks = _parse_relevance_results(results or [])

    data["relevance"] = [x[0] for x in checks]
    data["reasoning"] = [x[1] for x in checks]
    data["relevance"] = data["relevance"].astype("boolean")
    return data[data["relevance"]]


def reflect_task_status(
    user_query: str,
    research_results: str,
    model_id: str = config["models"]["performance_low"],
) -> tuple[bool | None, str | None]:
    """Evaluate if the research task is finished based on the query and results."""
    prompt = REFLECT_TASK.format(
        user_query=user_query,
        research_results=research_results,
    )

    json_schema = _prepare_json_schema(ReflectTask)
    response = llm_client.call_structured(
        prompt=prompt,
        model_id=model_id,
        temperature=config["temperature"]["low"],
        json_schema=json_schema,
    )

    if not response:
        return None, None

    parsed = _parse_json_response(response)
    if parsed:
        return parsed.get("finished"), parsed.get("reflection")

    return None, None


def create_final_report(
    user_query: str,
    final_docs: pd.DataFrame,
    model_id: str = config["models"]["performance_high"],
) -> tuple[str, dict]:
    """Generate a final research report from selected documents."""
    research_results = [
        DOCUMENT.format(
            title=row["title"],
            text=row["text"],
            date=row["date"],
            link=row["link"],
            analysis=row["analysis"],
        )
        for _, row in final_docs.iterrows()
    ]

    custom_logger.info_console(
        f"Creating final report for query: {user_query} with {len(research_results)} documents."
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"{config['app']['save_final_docs_to']}final_docs_{timestamp}.json", "w") as f:
        json.dump(research_results, f, indent=2)

    research_results_text = "\n\n".join(research_results)

    response, usage = llm_client.call_with_reasoning(
        prompt=RESEARCH_WRITER.format(
            user_query=user_query, research_results=research_results_text
        ),
        model_id=model_id,
        temperature=config["temperature"]["base"],
    )
    return response, usage