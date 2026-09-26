"""Register the fixture experiments into the platform registry + seed the daily
variant view tables the scoresheet layer reads.

Run after build_dataset. Idempotent.
"""
from __future__ import annotations

import json
import duckdb

from nl2sql.abtest import registry as reg

DB = "./abtest_data/abtest.duckdb"
GT = "./abtest_data/ground_truth.json"

SHARED_METRICS = ["users", "purchases", "add_to_cart", "gmv",
                  "purch_per_user", "gmv_per_user", "purch_lift", "gmv_tot_lift"]
SHARED_DIMS = ["slice_country", "slice_page", "overall_flag"]


def main() -> None:
    gt = json.load(open(GT))["experiments"]
    con = duckdb.connect(DB)
    con.execute("CREATE SCHEMA IF NOT EXISTS exp")
    # per-experiment scoresheet tables named by the platform convention
    for exp_id in gt:
        con.execute(
            f"CREATE OR REPLACE TABLE exp.{exp_id} AS "
            f"SELECT * FROM exp_scoresheet WHERE experiment_id = '{exp_id}'"
        )
    con.close()

    reg.register(
        DB, "checkout_flow_v2", "p13n",
        table_name="checkout_flow_v2",
        name="One-page checkout",
        started_on=gt["checkout_flow_v2"]["started_on"],
        ended_on=gt["checkout_flow_v2"]["ended_on"],
        variants=[v["id"] for v in gt["checkout_flow_v2"]["variants"]],
        metric_columns=SHARED_METRICS, dimension_columns=SHARED_DIMS,
        north_star_metric="gmv_per_user",
        guardrail_metric="purch_per_user",
        ship_threshold=gt["checkout_flow_v2"]["ship_threshold"],
        guardrail_tolerance=gt["checkout_flow_v2"]["guardrail_tolerance"],
    )
    reg.register(
        DB, "pricing_page_copy_v3", "p13n",
        table_name="pricing_page_copy_v3",
        name="Pricing page headline rewrite",
        started_on=gt["pricing_page_copy_v3"]["started_on"],
        ended_on=gt["pricing_page_copy_v3"]["ended_on"],
        variants=[v["id"] for v in gt["pricing_page_copy_v3"]["variants"]],
        metric_columns=SHARED_METRICS, dimension_columns=SHARED_DIMS,
        north_star_metric="gmv_per_user",
        guardrail_metric="purch_per_user",
        ship_threshold=gt["pricing_page_copy_v3"]["ship_threshold"],
        guardrail_tolerance=gt["pricing_page_copy_v3"]["guardrail_tolerance"],
    )
    reg.register(
        DB, "onboarding_email_v1", "onboarding",
        table_name="onboarding_email_v1",
        name="Day-zero onboarding email",
        started_on=gt["onboarding_email_v1"]["started_on"],
        ended_on=gt["onboarding_email_v1"]["ended_on"],
        variants=[v["id"] for v in gt["onboarding_email_v1"]["variants"]],
        metric_columns=SHARED_METRICS, dimension_columns=SHARED_DIMS,
        north_star_metric="gmv_per_user",
        guardrail_metric="purch_per_user",
        ship_threshold=gt["onboarding_email_v1"]["ship_threshold"],
        guardrail_tolerance=gt["onboarding_email_v1"]["guardrail_tolerance"],
    )

    print("domains:", reg.domains(DB))
    for d in reg.domains(DB):
        for e in reg.experiments_for_domain(DB, d):
            r = reg.resolve(DB, d, e["experiment_id"])
            print(f"  {d}/{e['experiment_id']}: table={r['table_name']} "
                  f"exists={r['table_exists']} north_star={r['north_star_metric']}")


if __name__ == "__main__":
    main()
