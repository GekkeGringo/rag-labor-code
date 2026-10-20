from datetime import datetime

from airflow.sdk import dag, task


@dag(schedule="@weekly", start_date=datetime(2026, 1, 1), catchup=False, tags=["rag"])
def labor_code_index():
    @task.short_circuit
    def check_source() -> bool:
        from rag_labor_code.pipeline import SOURCE, source_changed

        return source_changed(SOURCE)

    @task
    def prepare_data() -> int:
        from rag_labor_code.pipeline import prepare

        return len(prepare())

    @task
    def build_index(n_articles: int) -> str:
        from rag_labor_code.pipeline import build_and_publish, load_articles

        return build_and_publish(load_articles())

    check_source() >> build_index(prepare_data())


labor_code_index()