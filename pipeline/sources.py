import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests

from .manifest import Manifest, calculate_hash


DEFAULT_TIMEOUT = 30
DEFAULT_RATE_LIMIT_DELAY = 1.0
DEFAULT_MAX_RETRIES = 3


@dataclass
class RateLimitPolicy:
    delay_seconds: float = DEFAULT_RATE_LIMIT_DELAY
    max_retries: int = DEFAULT_MAX_RETRIES


def _sleep(delay_seconds: float) -> None:
    if delay_seconds > 0:
        time.sleep(delay_seconds)


def _request_json(
    session: requests.Session,
    url: str,
    logger,
    params: Optional[Dict[str, Any]] = None,
    policy: Optional[RateLimitPolicy] = None,
    timeout: int = DEFAULT_TIMEOUT,
    allow_404: bool = False,
) -> Optional[Dict[str, Any]]:
    policy = policy or RateLimitPolicy()
    for attempt in range(1, policy.max_retries + 1):
        try:
            response = session.get(url, params=params, timeout=timeout)
            if response.status_code in {429} or response.status_code >= 500:
                logger.error(
                    "Transient error from source.",
                    url=url,
                    status=response.status_code,
                    attempt=attempt,
                )
                _sleep(policy.delay_seconds * attempt)
                continue
            if response.status_code == 404:
                if not allow_404:
                    body_preview = (response.text or "").strip().replace("\n", " ")[:300]
                    logger.error(
                        "Client error from source.",
                        url=url,
                        status=response.status_code,
                        attempt=attempt,
                        content_type=response.headers.get("Content-Type"),
                        server=response.headers.get("Server"),
                        cache=response.headers.get("X-Cache"),
                        drupal_cache=response.headers.get("X-Drupal-Cache"),
                        request_id=response.headers.get("X-Request-Id"),
                        body_preview=body_preview,
                    )
                break
            if 400 <= response.status_code < 500:
                body_preview = (response.text or "").strip().replace("\n", " ")[:300]
                logger.error(
                    "Client error from source.",
                    url=url,
                    status=response.status_code,
                    attempt=attempt,
                    content_type=response.headers.get("Content-Type"),
                    server=response.headers.get("Server"),
                    cache=response.headers.get("X-Cache"),
                    drupal_cache=response.headers.get("X-Drupal-Cache"),
                    request_id=response.headers.get("X-Request-Id"),
                    body_preview=body_preview,
                )
                _sleep(policy.delay_seconds * attempt)
                continue
            response.raise_for_status()
            _sleep(policy.delay_seconds)
            return response.json()
        except requests.RequestException as exc:
            logger.error("Request failed.", url=url, error=str(exc), attempt=attempt)
            _sleep(policy.delay_seconds * attempt)
    return None


def _request_text(
    session: requests.Session,
    url: str,
    logger,
    policy: Optional[RateLimitPolicy] = None,
    timeout: int = DEFAULT_TIMEOUT,
    allow_404: bool = False,
) -> Optional[str]:
    policy = policy or RateLimitPolicy()
    for attempt in range(1, policy.max_retries + 1):
        try:
            response = session.get(url, timeout=timeout)
            if response.status_code in {429} or response.status_code >= 500:
                logger.error(
                    "Transient error from source.",
                    url=url,
                    status=response.status_code,
                    attempt=attempt,
                )
                _sleep(policy.delay_seconds * attempt)
                continue
            if response.status_code == 404:
                if not allow_404:
                    body_preview = (response.text or "").strip().replace("\n", " ")[:300]
                    logger.error(
                        "Client error from source.",
                        url=url,
                        status=response.status_code,
                        attempt=attempt,
                        content_type=response.headers.get("Content-Type"),
                        server=response.headers.get("Server"),
                        cache=response.headers.get("X-Cache"),
                        drupal_cache=response.headers.get("X-Drupal-Cache"),
                        request_id=response.headers.get("X-Request-Id"),
                        body_preview=body_preview,
                    )
                break
            if 400 <= response.status_code < 500:
                body_preview = (response.text or "").strip().replace("\n", " ")[:300]
                logger.error(
                    "Client error from source.",
                    url=url,
                    status=response.status_code,
                    attempt=attempt,
                    content_type=response.headers.get("Content-Type"),
                    server=response.headers.get("Server"),
                    cache=response.headers.get("X-Cache"),
                    drupal_cache=response.headers.get("X-Drupal-Cache"),
                    request_id=response.headers.get("X-Request-Id"),
                    body_preview=body_preview,
                )
                _sleep(policy.delay_seconds * attempt)
                continue
            response.raise_for_status()
            _sleep(policy.delay_seconds)
            return response.text
        except requests.RequestException as exc:
            logger.error("Request failed.", url=url, error=str(exc), attempt=attempt)
            _sleep(policy.delay_seconds * attempt)
    return None


def _is_drupal_core_11(constraint: str) -> bool:
    if not constraint:
        return False
    return re.search(r"(^|[^\d])\^?11(\.|$)", constraint) is not None


def _extract_machine_name(node: Dict[str, Any]) -> Optional[str]:
    for key in ("field_project_machine_name", "machine_name", "project_machine_name"):
        value = node.get(key)
        if isinstance(value, dict) and "value" in value:
            return value["value"]
        if isinstance(value, str):
            return value
    return None


def _extract_default_branch(node: Dict[str, Any]) -> Optional[str]:
    value = node.get("field_project_default_branch")
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    if isinstance(value, str):
        return value
    return None


def _iter_drupal_project_nodes(
    session: requests.Session,
    endpoint: str,
    logger,
    policy: RateLimitPolicy,
    max_pages: int = 1,
    limit: Optional[int] = None,
    filters: Optional[Dict[str, Any]] = None,
    sort: str = "changed",
    direction: str = "desc",
) -> Iterable[Dict[str, Any]]:
    for page in range(max_pages):
        params = {
            "type": "project_module",
            "status": 1,
            "sort": sort,
            "direction": direction,
        }
        # Only pass valid filters to node.json API
        if filters:
            for k, v in filters.items():
                if k in {"field_project_machine_name", "nid"}:
                    params[k] = v
        if page > 0:
            params["page"] = page
        if limit:
            params["limit"] = limit
        payload = _request_json(
            session,
            endpoint,
            logger,
            params=params,
            policy=policy,
        )
        if not payload or "list" not in payload:
            break
        entries = payload.get("list", [])
        if not entries:
            break
        for node in entries:
            yield node


def _discover_drupal_projects(
    session: requests.Session,
    endpoint: str,
    logger,
    policy: RateLimitPolicy,
    max_pages: int = 1,
    limit: Optional[int] = None,
    filters: Optional[Dict[str, Any]] = None,
    sort: str = "changed",
    direction: str = "desc",
) -> List[Dict[str, Any]]:
    return list(
        _iter_drupal_project_nodes(
            session,
            endpoint,
            logger,
            policy,
            max_pages=max_pages,
            limit=limit,
            filters=filters,
            sort=sort,
            direction=direction,
        )
    )


def _fetch_composer_constraint(
    session: requests.Session,
    logger,
    machine_name: str,
    branches: List[str],
    policy: RateLimitPolicy,
) -> Optional[Dict[str, Any]]:
    for branch in branches:
        # Use cleaner raw URL pattern (works across more GitLab versions/proxies)
        url = (
            f"https://git.drupalcode.org/project/{machine_name}/raw/"
            f"{branch}/composer.json"
        )
        text = _request_text(session, url, logger, policy=policy, allow_404=True)
        if text:
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                logger.error("Invalid composer.json content.", url=url)
                return None
            requirement = data.get("require", {}).get("drupal/core")
            return {"constraint": requirement, "branch": branch, "url": url}
    return None


def _build_curated_sources(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    drupal_core = config.get("sources", {}).get("drupal_core", {})
    return [
        {
            "id": "drupal_core",
            "type": drupal_core.get("type", "git"),
            "url": drupal_core.get("url"),
            "ref": drupal_core.get("branch"),
            "description": "Drupal core repository",
        },
        {
            "id": "drupal_docs",
            "type": "http",
            "url": "https://www.drupal.org/docs",
            "ref": None,
            "description": "Drupal official documentation",
        },
        {
            "id": "drupal_api",
            "type": "http",
            "url": "https://api.drupal.org/api/drupal",
            "ref": None,
            "description": "Drupal API reference",
        },
        {
            "id": "symfony_docs",
            "type": "http",
            "url": "https://symfony.com/doc/7.0/index.html",
            "ref": None,
            "description": "Symfony 7 documentation",
        },
        {
            "id": "drupal_security",
            "type": "http",
            "url": "https://www.drupal.org/security",
            "ref": None,
            "description": "Drupal security advisories",
        },
    ]


def _write_sources_manifest(
    path: Path,
    seed: Optional[int],
    curated_sources: List[Dict[str, Any]],
    drupal_sources: List[Dict[str, Any]],
    policy: RateLimitPolicy,
    metrics: Dict[str, Any],
) -> None:
    manifest = {
        "stage": "sources",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": seed,
        "rate_limit": {
            "delay_seconds": policy.delay_seconds,
            "max_retries": policy.max_retries,
        },
        "metrics": metrics,
        "sources": {
            "curated": curated_sources,
            "drupal_org_projects": drupal_sources,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(manifest, handle, indent=2)


def run_sources_stage(config: Dict[str, Any], logger, root: Path) -> int:
    sources_dir = root / config.get("directories", {}).get("sources", "sources")
    seed = config.get("seed")
    policy = RateLimitPolicy()
    session = requests.Session()
    drupal_project_config = config.get("sources", {}).get("drupal_projects", {})
    user_agent = drupal_project_config.get("user_agent") or "DrupalGym/1.0"
    headers = {
        "User-Agent": user_agent,
        "Accept": "application/json",
    }
    from_header = drupal_project_config.get("from")
    referer = drupal_project_config.get("referer")
    extra_headers = drupal_project_config.get("headers", {})
    if from_header:
        headers["From"] = from_header
    if referer:
        headers["Referer"] = referer
    if isinstance(extra_headers, dict):
        headers.update(extra_headers)
    session.headers.update(headers)
    curated_sources = _build_curated_sources(config)

    drupal_endpoint = drupal_project_config.get("endpoint")
    discovered_projects: List[Dict[str, Any]] = []
    filtered_projects: List[Dict[str, Any]] = []
    if drupal_endpoint:
        logger.info("Discovering Drupal.org projects.", endpoint=drupal_endpoint)
        max_pages = int(drupal_project_config.get("max_pages", 5))
        limit = drupal_project_config.get("page_size") or drupal_project_config.get("limit") or 100
        filters = drupal_project_config.get("filters", {})
        sort = drupal_project_config.get("sort", "changed")
        direction = drupal_project_config.get("direction", "desc")
        discovered_projects = _discover_drupal_projects(
            session,
            drupal_endpoint,
            logger,
            policy,
            max_pages=max_pages,
            limit=limit,
            filters=filters,
            sort=sort,
            direction=direction,
        )
        
        # Helper to calculate total usage for sorting
        def get_total_usage(node):
            usage = node.get("project_usage", {})
            if not isinstance(usage, dict):
                return 0
            return sum(int(v) for v in usage.values() if str(v).isdigit())

        # Sort discovered projects by usage locally to find the most popular ones
        discovered_projects.sort(key=get_total_usage, reverse=True)
        
        logger.info(
            "Discovered and sorted Drupal.org projects.",
            count=len(discovered_projects),
        )

    for project in discovered_projects:
        if len(filtered_projects) >= 100:
            break
        machine_name = _extract_machine_name(project)
        if not machine_name:
            continue
        default_branch = _extract_default_branch(project)
        candidate_branches = []
        if default_branch:
            candidate_branches.append(default_branch)
        
        # Focused list of common branch patterns for modern Drupal
        # Prefer semantic versioning patterns first for speed
        common_patterns = ["11.x", "11.0.x", "1.x", "1.0.x", "2.x", "3.x", "8.x-1.x", "main", "master"]
        for p in common_patterns:
            if p not in candidate_branches:
                candidate_branches.append(p)
                
        composer_info = _fetch_composer_constraint(
            session, logger, machine_name, candidate_branches, policy
        )
        if not composer_info:
            continue
        constraint = composer_info.get("constraint")
        if not _is_drupal_core_11(str(constraint)):
            continue
        filtered_projects.append(
            {
                "id": machine_name,
                "type": "git",
                "url": f"https://git.drupalcode.org/project/{machine_name}.git",
                "ref": composer_info.get("branch"),
                "composer_url": composer_info.get("url"),
                "core_constraint": constraint,
            }
        )

    metrics = {
        "projects_discovered": len(discovered_projects),
        "projects_filtered": len(filtered_projects),
        "curated_sources": len(curated_sources),
    }
    logger.metric("projects_discovered", len(discovered_projects))
    logger.metric("projects_filtered", len(filtered_projects))
    logger.metric("curated_sources", len(curated_sources))

    manifest_path = sources_dir / "manifest.json"
    _write_sources_manifest(
        manifest_path,
        seed,
        curated_sources,
        filtered_projects,
        policy,
        metrics,
    )

    sources_manifest = Manifest("sources", sources_dir)
    sources_manifest.set_metrics(metrics)
    manifest_hash = calculate_hash(manifest_path)
    sources_manifest.add_output(
        "sources_manifest",
        str(manifest_path.relative_to(root)),
        manifest_hash,
    )
    sources_manifest.save(root / "manifests" / "sources_manifest.json")

    logger.info("Wrote sources manifest.", path=str(manifest_path))
    return 0
