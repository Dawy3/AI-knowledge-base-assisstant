"""
Metrics Collection using Prometheus.

FOCUS: Retrieval + generation + cost metrics
MUST: Recall@10, NDCG@10, Hit Rate
MUST: P50/P99 latency, token usage
TARGET: Recall@10 >80%, P99 <200ms
"""

from prometheus_client import Counter, Histogram, Gauge, Info


# Request metrics
REQUESTS_TOTAL = Counter(
    "rag_requests_total",
    "Total requests",
    ["endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "rag_request_latency_seconds",
    "Request latency",
    ["endpoint"],
    buckets=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
)

# Retrieval metrics
RETRIEVAL_LATENCY = Histogram(
    "rag_retrieval_latency_seconds",
    "Retrieval latency",
    ["search_type"],  # vector, bm25, hybrid
    buckets=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
)

RETRIEVAL_RESULTS = Histogram(
    "rag_retrieval_results_count",
    "Number of results returned",
    ["search_type"],
    buckets=[0, 1, 5, 10, 20, 50, 100],
)

# Generation metrics
GENERATION_LATENCY = Histogram(
    "rag_generation_latency_seconds",
    "LLM generation latency",
    ["model"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
)

TOKENS_USED = Counter(
    "rag_tokens_total",
    "Total tokens used",
    ["model", "type"],  # type: prompt, completion
)

# Cache metrics
CACHE_HITS = Counter(
    "rag_cache_hits_total",
    "Cache hits",
    ["cache_type"],  # semantic, embedding
)

CACHE_MISSES = Counter(
    "rag_cache_misses_total",
    "Cache misses",
    ["cache_type"],
)

# Routing metrics
ROUTING_DECISIONS = Counter(
    "rag_routing_total",
    "Routing decisions",
    ["route"],  # local, gpt4, gpt35
)

# Quality metrics (updated periodically from evaluation)
RECALL_AT_10 = Gauge(
    "rag_recall_at_10",
    "Current Recall@10 score",
)

NDCG_AT_10 = Gauge(
    "rag_ndcg_at_10",
    "Current NDCG@10 score",
)

# Cost tracking
ESTIMATED_COST = Counter(
    "rag_estimated_cost_dollars",
    "Estimated cost in dollars",
    ["model"],
)


class MetricsCollector:
    """
    Simple wrapper for recording metrics.
    
    Usage:
        metrics = MetricsCollector()
        
        with metrics.track_request("chat"):
            # handle request
            pass
        
        metrics.record_retrieval(latency=0.05, results=10, search_type="hybrid")
        metrics.record_generation(latency=1.2, model="gpt-4", tokens=500)
    """
    
    def track_request(self, endpoint: str):
        """Context manager for tracking request latency."""
        return REQUEST_LATENCY.labels(endpoint=endpoint).time()
    
    def record_request(self, endpoint: str, status: str = "success"):
        """Record request count."""
        REQUESTS_TOTAL.labels(endpoint=endpoint, status=status).inc()
    
    def record_retrieval(
        self,
        latency: float,
        results: int,
        search_type: str = "hybrid",
    ):
        """Record retrieval metrics."""
        RETRIEVAL_LATENCY.labels(search_type=search_type).observe(latency)
        RETRIEVAL_RESULTS.labels(search_type=search_type).observe(results)
    
    def record_generation(
        self,
        latency: float,
        model: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ):
        """Record generation metrics."""
        GENERATION_LATENCY.labels(model=model).observe(latency)
        
        if prompt_tokens:
            TOKENS_USED.labels(model=model, type="prompt").inc(prompt_tokens)
        if completion_tokens:
            TOKENS_USED.labels(model=model, type="completion").inc(completion_tokens)
    
    def record_cache(self, cache_type: str, hit: bool):
        """Record cache hit/miss."""
        if hit:
            CACHE_HITS.labels(cache_type=cache_type).inc()
        else:
            CACHE_MISSES.labels(cache_type=cache_type).inc()
    
    def record_routing(self, route: str):
        """Record routing decision."""
        ROUTING_DECISIONS.labels(route=route).inc()
    
    def record_cost(self, model: str, cost: float):
        """Record estimated cost."""
        ESTIMATED_COST.labels(model=model).inc(cost)
    
    def update_quality_metrics(self, recall: float, ndcg: float):
        """Update quality metrics from evaluation."""
        RECALL_AT_10.set(recall)
        NDCG_AT_10.set(ndcg)
