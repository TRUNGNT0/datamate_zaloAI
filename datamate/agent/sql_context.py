"""Build SQL context from adapter-produced DuckDB and catalog, never answer keys."""
import hashlib
import re
from pathlib import Path

from datamate.ingestion.processing.pipeline import IngestionPipeline
from datamate.ingestion.catalog import PIPELINE_VERSION, CatalogService
from datamate.ingestion.sources.adapters.common import TabularLayout
from datamate.text import norm
from datamate.tools.sql import SQLExecutor
from datamate.tools.timeutil import describe_period


class SQLContextBuilder:
    def __init__(self, cfg):
        self.cfg = cfg
        self.prepared = {}

    def prepare(self, split, business_id):
        root = Path(split.root).resolve()
        key = (str(root), business_id)
        if key in self.prepared:
            return self.prepared[key]
        business_dir = root / "data" / business_id
        if business_dir.resolve().parent != root / "data":
            raise ValueError("business_id không hợp lệ")
        output = Path(self.cfg.database_dir) / business_id / "current"
        database = output / "data.duckdb"
        manifest = (Path(self.cfg.manifest_dir) if self.cfg.manifest_dir else root / "manifests") / f"{business_id}.json"
        if self.cfg.require_prepared_database and not manifest.is_file():
            raise ValueError(f"Missing prepared manifest: {manifest}")
        manifest = manifest if manifest.exists() else None
        overrides = self.cfg.ingestion_overrides.get(business_id, split.ingestion_overrides.get(business_id))
        signature = IngestionPipeline.input_fingerprint(business_dir, rulebook=split.rulebook, manifest_path=manifest, overrides=overrides)
        catalog = None
        if database.exists() and not self.cfg.rebuild_database:
            import duckdb
            try:
                catalog = CatalogService.read_catalog(database)
            except (duckdb.Error, ValueError):
                pass  # Old/incomplete cache: rebuild from sources.
        if (catalog is None or catalog.get("pipeline_version") != PIPELINE_VERSION
                or catalog.get("input_fingerprint") != signature):
            if self.cfg.require_prepared_database:
                raise ValueError(f"Database missing or stale: {database}; run prepare_datasets() first")
            catalog = IngestionPipeline.ingest(business_dir, output, manifest,
                rulebook=split.rulebook, overrides=overrides)
        if catalog["business_id"] != business_id:
            raise ValueError("Catalog không thuộc business đang hỏi")
        tables = []
        with SQLExecutor.connect(database, split.calendar) as connection:
            # Point 2: Check and apply cached canonical views & debt layer from DataMate
            views_dir = Path(getattr(self.cfg, "views_from", "configs/public_v9") or "configs/public_v9")
            if getattr(self.cfg, "views", "cached") != "none" and views_dir.is_dir():
                v_file = views_dir / f"_views_{business_id}.json"
                d_file = views_dir / f"_debt_{business_id}.json"
                try:
                    has_v_sales = bool(connection.execute("SELECT 1 FROM information_schema.tables WHERE table_name = 'v_sales'").fetchone())
                except Exception:
                    has_v_sales = False
                if not has_v_sales and v_file.is_file():
                    try:
                        import json
                        from datamate import views
                        v_sql = json.loads(v_file.read_text(encoding="utf-8"))
                        views.build(connection, v_sql)
                    except Exception as err:
                        print(f"[{business_id}] Bỏ qua tải v_sales từ {v_file.name}: {err}", flush=True)
                if getattr(self.cfg, "debt_views", True) and d_file.is_file():
                    try:
                        has_debt = bool(connection.execute("SELECT 1 FROM information_schema.tables WHERE table_name = 'v_debt_orders'").fetchone())
                    except Exception:
                        has_debt = False
                    if not has_debt:
                        try:
                            import json
                            from datamate import debts
                            d_sqls = json.loads(d_file.read_text(encoding="utf-8"))
                            if d_sqls:
                                debts.build(connection, d_sqls)
                        except Exception as err:
                            print(f"[{business_id}] Bỏ qua tải debt layer từ {d_file.name}: {err}", flush=True)

            for table in catalog["tables"]:
                name = TabularLayout.quote_identifier(table["physical_name"])
                _, rows = SQLExecutor.query(connection, f"SELECT * FROM {name} LIMIT 3")
                columns = []
                for column in table["columns"]:
                    profile = column.get("profile", {})
                    row_count = profile.get("row_count", table["row_count"])
                    null_count = profile.get("null_count", 0)
                    parse_failed_count = profile.get("parse_failed_count", 0)
                    source_null_count = max(0, null_count - parse_failed_count)
                    present_count = max(0, row_count - source_null_count)
                    item = {k: column.get(k) for k in (
                        "physical_name", "source_name", "duckdb_type", "description",
                        "semantic_name", "semantic_status", "unit_basis", "aggregation", "time_role", "profile", "conversion")}
                    item["completeness"] = {
                        "row_count": row_count,
                        "source_null_count": source_null_count,
                        "parse_failed_count": parse_failed_count,
                        "typed_value_count": max(0, row_count - null_count),
                        "parse_success_rate": (
                            round((present_count - parse_failed_count) / present_count, 4)
                            if present_count else None
                        ),
                        "status": "parse_failures" if parse_failed_count else "no_parse_failures",
                    }
                    col = TabularLayout.quote_identifier(column["physical_name"])
                    if (column.get("time_role") or re.search(r"ngay|date|thoi_gian", column["physical_name"])):
                        year = re.search(r"\b(?:19|20)\d{2}\b", " ".join(table["source_objects"]))
                        expression = (f"CAST({col} AS DATE)" if column["duckdb_type"] in ("DATE", "TIMESTAMP") else
                                      f"vn_date_year(CAST({col} AS VARCHAR), {year[0]})" if year
                                      else f"vn_date(CAST({col} AS VARCHAR))")
                        _, span = SQLExecutor.query(connection, f"SELECT min({expression}), max({expression}) FROM {name}")
                        item["date_range"] = span[0]
                    if re.search(r"trang_thai|status|ket_qua|tinh_trang", column["physical_name"]):
                        _, counts = SQLExecutor.query(connection,
                            f"SELECT {col}, count(*) AS n FROM {name} GROUP BY 1 ORDER BY 2 DESC LIMIT 100")
                        item["values"] = counts
                        item["revenue_status_values"] = [
                            {"value": value, "count": count,
                             "classification": SQLContextBuilder.classify_revenue_status(value)}
                            for value, count in counts
                        ]
                    columns.append(item)
                tables.append({"name": table["physical_name"], "sources": table["source_files"],
                               "source_objects": table["source_objects"], "role": table.get("table_role"),
                               "grain": table.get("grain"), "row_count": table["row_count"],
                               "columns": columns, "sample_rows": rows,
                               "lineage": table.get("lineage"), "issue_counts": table.get("issue_counts", {})})
        readme_path = business_dir / "README.md"
        context = {"tables": tables, "relationships": catalog.get("relationships", []),
                   "facts": catalog.get("facts", []), "unresolved": catalog.get("unresolved", []),
                   "quality": catalog.get("quality", {}),
                   "organization": self.organization_context(catalog),
                   "semantic_views": catalog.get("semantic_views", []),
                   "rulebook": split.rulebook,
                   "readme": readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""}
        if split.calendar is not None:
            context["calendar"] = {"name": "calendar", "columns": list(split.calendar.columns),
                                   "source": "calendar.csv",
                                   "sha256": hashlib.sha256(split.calendar.to_json(
                                       orient="split", date_format="iso").encode()).hexdigest()}
        with database.open("rb") as stream:
            fingerprint = hashlib.file_digest(stream, "sha256").hexdigest()
        self.prepared[key] = database, context, fingerprint
        return self.prepared[key]

    @staticmethod
    def organization_context(catalog):
        """Send applied view mappings/grain, not installed SQL or rejected candidates."""
        organization = catalog.get("organization", {})
        result = {k: organization.get(k) for k in ("policy", "policy_version", "relationships")}
        for key in ("transformations", "models"):
            result[key] = [{k: v for k, v in op.items()
                            if k not in ("installed_sql", "definition_fingerprint", "queries")}
                           for op in organization.get(key, []) if op["status"] == "applied"]
        return result

    @staticmethod
    def classify_revenue_status(value):
        status = norm(str(value))
        if any(term in status for term in ("khong giao", "giao that bai", "giao khong thanh cong")):
            return "exclude_failed_delivery"
        if any(term in status for term in ("tra hang", "doi tra", "returned")):
            return "exclude_returned"
        if any(term in status.split() for term in ("huy", "cancel", "cancelled", "canceled", "void")):
            return "exclude_cancelled"
        if any(term in status for term in ("dang van chuyen", "dang giao", "in transit")):
            return "include_in_transit"
        if any(term in status for term in ("da giao", "hoan thanh", "delivered", "completed")):
            return "include_completed"
        return "unknown"

    @staticmethod
    def question_context(context, original, rewritten, as_of, calendar, db=None, bank=None, business_id=None):
        from datamate.hints import question_hints
        meta = {"tables": context.get("tables", [])}
        rulebook_txt = context.get("rulebook", "")
        dynamic_hints = question_hints(original, str(as_of), db=db, meta=meta, rulebook=rulebook_txt)
        time_hint = describe_period(rewritten, as_of, calendar)
        
        res = {
            **context,
            "original_question": original,
            "question": rewritten,
            "as_of_date": str(as_of),
            "time_hint": time_hint,
            "hints": dynamic_hints,
        }
        if bank is not None and business_id:
            try:
                exemplars = bank.retrieve(original, business_id, k=4)
                if exemplars:
                    res["exemplars"] = exemplars
            except Exception:
                pass
        return res
