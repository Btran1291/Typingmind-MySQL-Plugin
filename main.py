import logging
import traceback
from fastapi import FastAPI, Depends, HTTPException, Body
from sqlalchemy import create_engine, MetaData, Table, select, and_, or_, text, func
from sqlalchemy.orm import sessionmaker, Session, aliased
from sqlalchemy.sql import operators
from sqlalchemy.exc import IntegrityError, SQLAlchemyError, OperationalError, ProgrammingError
from threading import Lock
from typing import List, Optional, Union, Dict, Any
from models import (
    Query, Filters, FilterCondition, Join, OrderBy, Insert, Update, Delete,
    Aggregation, Count, BatchInsert, BatchUpdateItem, BatchUpdate,
    BatchDeleteItem, BatchDelete, DatabaseCredentials
)
from fastapi.middleware.cors import CORSMiddleware

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Restrict origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Operator Map ---
OPERATOR_MAP = {
    "eq": operators.eq,
    "neq": operators.ne,
    "gt": operators.gt,
    "gte": operators.ge,
    "lt": operators.lt,
    "lte": operators.le,
    "like": operators.like_op,
    "in": operators.in_op,
    "nin": operators.notin_op,
}

_engine_cache = {}
_engine_cache_lock = Lock()

def get_engine_metadata_and_session_factory(credentials: DatabaseCredentials, force_re_reflect: bool = False):
    """
    Given DatabaseCredentials, return a tuple of (engine, metadata, SessionLocal).
    Caches engines to avoid creating multiple engines for same credentials.
    If force_re_reflect is True, it will clear and re-reflect metadata for an existing engine.
    """
    log_connection_url_safe = (
        f"mysql+mysqlconnector://{credentials.mysql_user}:********"
        f"@{credentials.mysql_host}"
        f"/{credentials.mysql_database}"
    )
    connection_url = (
        f"mysql+mysqlconnector://{credentials.mysql_user}:{credentials.mysql_password}"
        f"@{credentials.mysql_host}"
        f"/{credentials.mysql_database}"
    )

    with _engine_cache_lock:
        if connection_url in _engine_cache and not force_re_reflect:
            engine, metadata, SessionLocal = _engine_cache[connection_url]
            logger.debug(f"Using cached engine for {log_connection_url_safe}")
        else:
            try:
                if connection_url in _engine_cache and force_re_reflect:
                    logger.info(f"Forcing re-reflection for existing engine: {log_connection_url_safe}")
                    engine, metadata, SessionLocal = _engine_cache[connection_url]
                    metadata.clear()
                    metadata.reflect(bind=engine)
                else:
                    logger.info(f"Creating new engine for {log_connection_url_safe}")
                    engine = create_engine(
                        connection_url,
                        pool_pre_ping=True,
                        pool_size=5,
                        max_overflow=10,
                        pool_recycle=3600,
                        connect_args={"connect_timeout": 10}
                    )
                    logger.debug("Engine created. Attempting to reflect metadata...")
                    metadata = MetaData()
                    metadata.reflect(bind=engine)
                    logger.debug("Metadata reflected successfully. Creating sessionmaker...")
                    SessionLocal = sessionmaker(bind=engine)
                _engine_cache[connection_url] = (engine, metadata, SessionLocal)
                logger.info(f"Engine setup/re-reflection complete for {log_connection_url_safe}")
            except OperationalError as e:
                logger.error(f"OperationalError during database connection to {log_connection_url_safe}: {e}", exc_info=True)
                error_detail = str(e)
                if "Access denied" in error_detail:
                     raise HTTPException(status_code=401, detail="Database authentication failed. Check credentials.")
                elif "Unknown database" in error_detail:
                     raise HTTPException(status_code=400, detail="Database not found. Check database name.")
                elif "Can't connect to MySQL server" in error_detail or "timed out" in error_detail or "Connection refused" in error_detail:
                     raise HTTPException(status_code=504, detail="Cannot connect to database server. Check host, port, and network connectivity. Is MySQL server running?")
                else:
                     raise HTTPException(status_code=500, detail=f"Database connection error: {error_detail}")
            except ProgrammingError as e:
                logger.error(f"ProgrammingError during database reflection for {log_connection_url_safe}: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Database schema reflection error: {e}")
            except SQLAlchemyError as e:
                logger.error(f"SQLAlchemyError during database setup for {log_connection_url_safe}: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Database setup error: {e}")
            except Exception as e:
                logger.error(f"An unexpected error occurred during database engine setup for {log_connection_url_safe}: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Internal server error during database setup: {e}")

    return engine, metadata, SessionLocal


def validate_table_exists(table_name: str, credentials: DatabaseCredentials, retry_count: int = 0):
    _, metadata, _ = get_engine_metadata_and_session_factory(credentials, force_re_reflect=(retry_count > 0))

    if table_name not in metadata.tables:
        if retry_count < 1:
            logger.warning(f"Table '{table_name}' not found in current metadata. Attempting to re-reflect schema.")
            validate_table_exists(table_name, credentials, retry_count=retry_count + 1)
        else:
            raise HTTPException(status_code=400, detail=f"Table '{table_name}' does not exist even after schema refresh. Please ensure the table exists and credentials are correct.")
    else:
        logger.info(f"Table '{table_name}' found in metadata.")


def validate_columns_exist(
    columns: List[str],
    alias_map: Dict[str, Table],
    main_table: Table,
    credentials: DatabaseCredentials,
    context: str = "columns"
):
    for col_name in columns:
        try:
            get_column_from_alias_map(col_name, alias_map, main_table, credentials)
        except HTTPException as e:
            raise HTTPException(status_code=e.status_code, detail=f"Column '{col_name}' in {context} validation failed: {e.detail}")
        except Exception as e:
             raise HTTPException(status_code=500, detail=f"Unexpected error validating column '{col_name}' in {context}: {e}")

def validate_filters_exist(filters: Optional[Filters], alias_map: Dict[str, Table], main_table: Table, credentials: DatabaseCredentials):
    if not filters:
        return
    for cond in filters.conditions:
        if isinstance(cond, Filters):
            validate_filters_exist(cond, alias_map, main_table, credentials)
        elif isinstance(cond, FilterCondition):
            validate_columns_exist([cond.column], alias_map, main_table, credentials, context="filter")

def validate_insert_data_columns(data: Dict[str, Any], table: Table):
    """
    Validate that all keys in the insert data dictionary exist as columns in the table.
    """
    for col_name in data.keys():
        if col_name not in table.c:
            raise HTTPException(status_code=400, detail=f"Column '{col_name}' in insert data not found in table '{table.name}'.")

def validate_update_data_columns(data: Dict[str, Any], table: Table):
    """
    Validate that all keys in the update data dictionary exist as columns in the table.
    """
    for col_name in data.keys():
        if col_name not in table.c:
            raise HTTPException(status_code=400, detail=f"Column '{col_name}' in update data not found in table '{table.name}'.")


def get_table(table_name: str, credentials: DatabaseCredentials) -> Table:
    try:
        validate_table_exists(table_name, credentials)
        _, metadata, _ = get_engine_metadata_and_session_factory(credentials)
        return metadata.tables[table_name]
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in get_table for '{table_name}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error when accessing table '{table_name}': {e}")


def get_aliased_table(table_name: str, alias: Optional[str], alias_map: Dict[str, Table], credentials: DatabaseCredentials) -> Table:
    if alias:
        if alias in alias_map:
            return alias_map[alias]
        base_table = get_table(table_name, credentials)
        aliased_table = aliased(base_table, name=alias)
        alias_map[alias] = aliased_table
        return aliased_table
    else:
        return get_table(table_name, credentials)

def get_column_from_alias_map(col_name: str, alias_map: Dict[str, Table], main_table: Table, credentials: DatabaseCredentials):
    parts = col_name.split(".")
    if len(parts) == 2:
        alias_or_table, column = parts
        if alias_or_table == main_table.name:
            if column not in main_table.c:
                raise HTTPException(status_code=400, detail=f"Column '{column}' not found in main table '{main_table.name}'.")
            return main_table.c[column]
        elif alias_or_table in alias_map:
            aliased_table = alias_map[alias_or_table]
            if column not in aliased_table.c:
                raise HTTPException(status_code=400, detail=f"Column '{column}' not found in alias/table '{alias_or_table}'.")
            return aliased_table.c[column]
        else:
            try:
                base_table_for_alias = get_table(alias_or_table, credentials)
                aliased_table = aliased(base_table_for_alias, name=alias_or_table)
                alias_map[alias_or_table] = aliased_table

                if column not in aliased_table.c:
                    raise HTTPException(status_code=400, detail=f"Column '{column}' not found in newly reflected alias/table '{alias_or_table}'.")
                return aliased_table.c[column]
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Unexpected error resolving column '{col_name}' after potential re-reflection: {e}")

    elif len(parts) == 1:
        column = parts[0]
        if column in main_table.c:
            return main_table.c[column]
        for aliased_table in alias_map.values():
            if column in aliased_table.c:
                return aliased_table.c[column]
        raise HTTPException(status_code=400, detail=f"Column '{column}' not found in main or joined tables.")
    else:
        raise HTTPException(status_code=400, detail=f"Invalid column name format: '{col_name}'")

def build_filter_condition_with_alias_map(condition: FilterCondition, alias_map: Dict[str, Table], main_table: Table, credentials: DatabaseCredentials):
    col = get_column_from_alias_map(condition.column, alias_map, main_table, credentials)
    op_func = OPERATOR_MAP.get(condition.operator)
    if op_func is None:
        raise HTTPException(status_code=400, detail=f"Unsupported operator: '{condition.operator}'.")
    val = condition.value
    if condition.operator in ("in", "nin") and not isinstance(val, list):
        raise HTTPException(status_code=400, detail=f"Operator '{condition.operator}' requires a list value.")
    if condition.operator == "like" and not isinstance(val, str):
        raise HTTPException(status_code=400, detail=f"Operator 'like' requires a string value.")
    if condition.operator == "like" and '%' not in val and '_' not in val:
        val = f"%{val}%"
    return op_func(col, val)

def build_filters_with_alias_map(filters: Filters, alias_map: Dict[str, Table], main_table: Table, credentials: DatabaseCredentials):
    if not filters or not filters.conditions:
        return text("1=1")
    conditions = []
    for cond in filters.conditions:
        if isinstance(cond, Filters):
            conditions.append(build_filters_with_alias_map(cond, alias_map, main_table, credentials))
        elif isinstance(cond, FilterCondition):
            conditions.append(build_filter_condition_with_alias_map(cond, alias_map, main_table, credentials))
        else:
            raise HTTPException(status_code=400, detail=f"Invalid filter condition type: {type(cond)}")
    if filters.logic == "and":
        return and_(*conditions)
    elif filters.logic == "or":
        return or_(*conditions)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported logic operator: '{filters.logic}'.")

def apply_joins(stmt, main_table, joins: List[Join], alias_map: Dict[str, Table], credentials: DatabaseCredentials):
    for join in joins:
        join_table = get_aliased_table(join.table, join.alias, alias_map, credentials)
        on_clauses = []
        for left_col, right_col in join.on.items():
            left_column = get_column_from_alias_map(left_col, alias_map, main_table, credentials)
            right_column = get_column_from_alias_map(right_col, alias_map, main_table, credentials)
            on_clauses.append(left_column == right_column)
        on_condition = and_(*on_clauses)
        join_type = join.type or "inner"
        if join_type == "inner":
            stmt = stmt.join(join_table, on_condition)
        elif join_type == "left":
            stmt = stmt.outerjoin(join_table, on_condition)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported join type: {join_type}")
        if join.columns:
            validate_columns_exist(join.columns, alias_map, join_table, credentials, context=f"join '{join.alias or join.table}' columns")
            cols = [join_table.c[col] for col in join.columns]
            stmt = stmt.add_columns(*cols)
        if join.filters:
            validate_filters_exist(join.filters, alias_map, join_table, credentials)
            filter_condition = build_filters_with_alias_map(join.filters, alias_map, join_table, credentials)
            stmt = stmt.where(filter_condition)
    return stmt


def apply_order_by(stmt, order_by_list: Optional[List[OrderBy]], alias_map: Dict[str, Table], main_table: Table, credentials: DatabaseCredentials):
    if not order_by_list:
        return stmt
    order_clauses = []
    for order in order_by_list:
        col = get_column_from_alias_map(order.column, alias_map, main_table, credentials)
        if order.direction.lower() == "asc":
            order_clauses.append(col.asc())
        elif order.direction.lower() == "desc":
            order_clauses.append(col.desc())
        else:
            raise HTTPException(status_code=400, detail=f"Invalid order direction: {order.direction}")
    return stmt.order_by(*order_clauses)


def build_aggregation_clauses(aggregations: List[Aggregation], alias_map: Dict[str, Table], main_table: Table, credentials: DatabaseCredentials):
    agg_clauses = []
    for agg in aggregations:
        if agg.function.lower() == "count" and agg.column is None:
            col = func.count()
        elif agg.column:
            col = get_column_from_alias_map(agg.column, alias_map, main_table, credentials)
            agg_func = getattr(func, agg.function.lower(), None)
            if agg_func is None:
                raise HTTPException(status_code=400, detail=f"Unsupported aggregation function: '{agg.function}'.")
            col = agg_func(col)
        else:
            raise HTTPException(status_code=400, detail=f"Column is required for aggregation function '{agg.function}'.")
        agg_clauses.append(col.label(agg.alias))
    return agg_clauses

def build_group_by_clauses(group_by_cols: List[str], alias_map: Dict[str, Table], main_table: Table, credentials: DatabaseCredentials):
    group_by_clauses = []
    for col_name in group_by_cols:
        col = get_column_from_alias_map(col_name, alias_map, main_table, credentials)
        group_by_clauses.append(col)
    return group_by_clauses

@app.get("/")
async def root():
    return {"message": "TypingMind MySQL API is running"}


@app.post("/schema")
async def get_schema(credentials: DatabaseCredentials):
    engine, metadata, _ = get_engine_metadata_and_session_factory(credentials, force_re_reflect=True)
    schema_data = []
    try:
        for table_name, table in metadata.tables.items():
            table_info = {
                "name": table_name,
                "columns": [],
                "primary_key": [],
                "foreign_keys": [],
                "indexes": []
            }
            for column in table.columns:
                column_info = {
                    "name": column.name,
                    "type": str(column.type),
                    "type_details": {
                        "python_type": column.type.python_type.__name__ if column.type.python_type else None,
                        "length": getattr(column.type, 'length', None),
                        "precision": getattr(column.type, 'precision', None),
                        "scale": getattr(column.type, 'scale', None),
                    },
                    "nullable": column.nullable,
                    "default": str(column.default.arg) if column.default else None,
                    "server_default": str(column.server_default.arg) if column.server_default else None,
                    "autoincrement": column.autoincrement
                }
                table_info["columns"].append(column_info)
                if column.primary_key:
                    table_info["primary_key"].append(column.name)
            for fk in table.foreign_keys:
                table_info["foreign_keys"].append({
                    "source_column": fk.parent.name,
                    "target_table": fk.column.table.name,
                    "target_column": fk.column.name
                })
            for index in table.indexes:
                index_info = {
                    "name": index.name,
                    "unique": index.unique,
                    "columns": [col.name for col in index.columns]
                }
                table_info["indexes"].append(index_info)
            schema_data.append(table_info)
        return {"schema": schema_data}
    except Exception as e:
        logger.error(f"Error retrieving database schema: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving database schema: {e}")

@app.post("/query")
async def run_query(query: Query = Body(...)):
    engine, _, SessionLocal = get_engine_metadata_and_session_factory(query.credentials)
    db = SessionLocal()
    try:
        validate_table_exists(query.table, query.credentials)
        main_table = get_table(query.table, query.credentials)

        alias_map = {main_table.name: main_table}
        if query.joins:
            for join in query.joins:
                validate_table_exists(join.table, query.credentials)
                get_aliased_table(join.table, join.alias, alias_map, query.credentials)

        if query.columns:
            validate_columns_exist(query.columns, alias_map, main_table, query.credentials, context="columns")

        validate_filters_exist(query.filters, alias_map, main_table, query.credentials)

        if query.group_by:
            validate_columns_exist(query.group_by, alias_map, main_table, query.credentials, context="group_by")

        if query.order_by:
            order_by_cols = [order.column for order in query.order_by]
            validate_columns_exist(order_by_cols, alias_map, main_table, query.credentials, context="order_by")

        if query.aggregations:
            agg_cols = [agg.column for agg in query.aggregations if agg.column]
            if agg_cols:
                validate_columns_exist(agg_cols, alias_map, main_table, query.credentials, context="aggregations")

        select_items = []
        is_aggregating = query.aggregations is not None and len(query.aggregations) > 0
        is_grouping = query.group_by is not None and len(query.group_by) > 0

        if is_aggregating or is_grouping:
            if is_grouping:
                group_by_clauses = build_group_by_clauses(query.group_by, alias_map, main_table, query.credentials)
                select_items.extend(group_by_clauses)
            if is_aggregating:
                agg_clauses = build_aggregation_clauses(query.aggregations, alias_map, main_table, query.credentials)
                select_items.extend(agg_clauses)
            if query.columns is not None and len(query.columns) > 0:
                raise HTTPException(status_code=400, detail="The 'columns' field cannot be used when 'aggregations' or 'group_by' are specified. Selected columns are defined by 'group_by' and 'aggregations'.")
        else:
            if query.columns:
                select_items = [get_column_from_alias_map(col, alias_map, main_table, query.credentials) for col in query.columns]
            else:
                select_items = [main_table]

        stmt = select(*select_items).select_from(main_table)

        if query.joins:
            stmt = apply_joins(stmt, main_table, query.joins, alias_map, query.credentials)

        if query.filters:
            stmt = stmt.where(build_filters_with_alias_map(query.filters, alias_map, main_table, query.credentials))

        if is_grouping:
             stmt = stmt.group_by(*group_by_clauses)

        if query.order_by:
            stmt = apply_order_by(stmt, query.order_by, alias_map, main_table, query.credentials)

        if query.limit is not None:
            stmt = stmt.limit(query.limit)
        if query.offset:
            stmt = stmt.offset(query.offset)

        result = db.execute(stmt)

        rows = [dict(zip(result.keys(), row)) for row in result]

        return {"data": rows}

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Unexpected error during query execution: {e}", exc_info=True)
        detail = str(e)
        if "1055" in detail or "GROUP BY" in detail:
             detail = "Query error: SELECT list contains nonaggregated column which is not included in GROUP BY clause."
        elif "Unknown column" in detail:
             detail = f"Query error: {detail}"
        elif "syntax error" in detail:
             detail = f"Query error: {detail}"
        else:
             detail = "An internal server error occurred."

        raise HTTPException(status_code=500, detail=detail)
    finally:
        if db:
            db.close()

@app.post("/insert")
async def insert_data(item: Insert):
    engine, _, SessionLocal = get_engine_metadata_and_session_factory(item.credentials)
    db = SessionLocal()
    try:
        validate_table_exists(item.table, item.credentials)
        table = get_table(item.table, item.credentials)
        validate_insert_data_columns(item.data, table)

        stmt = table.insert().values(**item.data)
        result = db.execute(stmt)
        db.commit()
        inserted_id = result.lastrowid if hasattr(result, 'lastrowid') else None
        logger.info(f"Inserted record into {item.table} with ID {inserted_id}")
        return {
            "message": f"Record successfully inserted into table '{item.table}'.",
            "inserted_id": inserted_id
        }
    except HTTPException as e:
        db.rollback()
        raise e
    except IntegrityError as e:
        db.rollback()
        logger.error(f"Integrity error during insert: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Database integrity error: {e.orig}")
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"SQLAlchemy error during insert: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    except Exception as e:
        db.rollback()
        logger.error(f"Unexpected error during insert: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
    finally:
        if db:
            db.close()

@app.post("/batch/insert")
async def batch_insert_data(item: BatchInsert):
    engine, _, SessionLocal = get_engine_metadata_and_session_factory(item.credentials)
    db = SessionLocal()
    try:
        validate_table_exists(item.table, item.credentials)
        table = get_table(item.table, item.credentials)
        if item.data:
             for row_data in item.data:
                  validate_insert_data_columns(row_data, table)

        if not item.data:
            logger.info(f"No data provided for batch insert into table '{item.table}'.")
            return {
                "message": f"No data provided for batch insert into table '{item.table}'.",
                "rows_inserted": 0
            }

        stmt = table.insert().values(item.data)
        result = db.execute(stmt)
        db.commit()
        logger.info(f"Batch inserted {result.rowcount} records into {item.table}")
        return {
            "message": f"Batch insert successful into table '{item.table}'.",
            "rows_inserted": result.rowcount
        }
    except HTTPException as e:
        db.rollback()
        raise e
    except IntegrityError as e:
        db.rollback()
        logger.error(f"Integrity error during batch insert: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Database integrity error: {e.orig}")
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"SQLAlchemy error during batch insert: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    except Exception as e:
        db.rollback()
        logger.error(f"Unexpected error during batch insert: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
    finally:
        if db:
            db.close()

@app.post("/update")
async def update_data(item: Update):
    engine, _, SessionLocal = get_engine_metadata_and_session_factory(item.credentials)
    db = SessionLocal()
    try:
        validate_table_exists(item.table, item.credentials)
        table = get_table(item.table, item.credentials)
        validate_update_data_columns(item.data, table)
        validate_filters_exist(item.filters, {}, table, item.credentials)

        stmt = table.update()
        if item.filters:
            filter_condition = build_filters_with_alias_map(item.filters, {}, table, item.credentials)
            stmt = stmt.where(filter_condition)
        stmt = stmt.values(**item.data)
        result = db.execute(stmt)
        db.commit()
        logger.info(f"Updated {result.rowcount} records in {item.table}")
        return {
            "message": f"Update successful on table '{item.table}'.",
            "rows_affected": result.rowcount
        }
    except HTTPException as e:
        db.rollback()
        raise e
    except Exception as e:
        db.rollback()
        logger.error(f"Unexpected error during update: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
    finally:
        if db:
            db.close()

@app.post("/batch/update")
async def batch_update_data(item: BatchUpdate):
    engine, _, SessionLocal = get_engine_metadata_and_session_factory(item.credentials)
    db = SessionLocal()
    total_rows_affected = 0
    try:
        validate_table_exists(item.table, item.credentials)
        table = get_table(item.table, item.credentials)
        if not item.updates:
            logger.info(f"No update specifications provided for table '{item.table}'.")
            return {
                "message": f"No update specifications provided for table '{item.table}'.",
                "total_rows_affected": 0
            }
        for update_item in item.updates:
             validate_update_data_columns(update_item.data, table)
             validate_filters_exist(update_item.filters, {}, table, item.credentials)

        for update_item in item.updates:
            stmt = table.update()
            filter_condition = build_filters_with_alias_map(update_item.filters, {}, table, item.credentials)
            stmt = stmt.where(filter_condition)
            stmt = stmt.values(**update_item.data)
            result = db.execute(stmt)
            total_rows_affected += result.rowcount

        db.commit()
        logger.info(f"Batch updated {total_rows_affected} records in {item.table}")
        return {
            "message": f"Batch update successful on table '{item.table}'.",
            "total_rows_affected": total_rows_affected
        }
    except HTTPException as e:
        db.rollback()
        raise e
    except IntegrityError as e:
        db.rollback()
        logger.error(f"Integrity error during batch update: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Database integrity error: {e.orig}")
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"SQLAlchemy error during batch update: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    except Exception as e:
        db.rollback()
        logger.error(f"Unexpected error during batch update: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
    finally:
        if db:
            db.close()

@app.post("/delete")
async def delete_data(item: Delete):
    engine, _, SessionLocal = get_engine_metadata_and_session_factory(item.credentials)
    db = SessionLocal()
    try:
        validate_table_exists(item.table, item.credentials)
        table = get_table(item.table, item.credentials)
        validate_filters_exist(item.filters, {}, table, item.credentials)

        stmt = table.delete()
        filter_condition = build_filters_with_alias_map(item.filters, {}, table, item.credentials)
        stmt = stmt.where(filter_condition)
        result = db.execute(stmt)
        db.commit()
        logger.info(f"Deleted {result.rowcount} records from {item.table}")
        return {
            "message": f"Delete successful on table '{item.table}'.",
            "rows_affected": result.rowcount
        }
    except HTTPException as e:
        db.rollback()
        raise e
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"SQLAlchemy error during delete: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    except Exception as e:
        db.rollback()
        logger.error(f"Unexpected error during delete: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
    finally:
        if db:
            db.close()


@app.post("/batch/delete")
async def batch_delete_data(item: BatchDelete):
    engine, _, SessionLocal = get_engine_metadata_and_session_factory(item.credentials)
    db = SessionLocal()
    total_rows_affected = 0
    try:
        validate_table_exists(item.table, item.credentials)
        table = get_table(item.table, item.credentials)
        if not item.deletions:
            logger.info(f"No delete specifications provided for table '{item.table}'.")
            return {
                "message": f"No delete specifications provided for table '{item.table}'.",
                "total_rows_affected": 0
            }
        for delete_item in item.deletions:
             validate_filters_exist(delete_item.filters, {}, table, item.credentials)

        for delete_item in item.deletions:
            stmt = table.delete()
            filter_condition = build_filters_with_alias_map(delete_item.filters, {}, table, item.credentials)
            stmt = stmt.where(filter_condition)
            result = db.execute(stmt)
            total_rows_affected += result.rowcount

        db.commit()
        logger.info(f"Batch deleted {total_rows_affected} records from {item.table}")
        return {
            "message": f"Batch delete successful on table '{item.table}'.",
            "total_rows_affected": total_rows_affected
        }
    except HTTPException as e:
        db.rollback()
        raise e
    except IntegrityError as e:
        db.rollback()
        logger.error(f"Integrity error during batch delete: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Database integrity error: {e.orig}")
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"SQLAlchemy error during batch delete: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    except Exception as e:
        db.rollback()
        logger.error(f"Unexpected error during batch delete: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
    finally:
        if db:
            db.close()

@app.post("/count")
async def count_data(item: Count):
    engine, _, SessionLocal = get_engine_metadata_and_session_factory(item.credentials)
    db = SessionLocal()
    try:
        validate_table_exists(item.table, item.credentials)
        table = get_table(item.table, item.credentials)
        validate_filters_exist(item.filters, {}, table, item.credentials)

        stmt = select(func.count()).select_from(table)
        if item.filters:
            filter_condition = build_filters_with_alias_map(item.filters, {}, table, item.credentials)
            stmt = stmt.where(filter_condition)

        result = db.execute(stmt)
        count_value = result.scalar()
        logger.info(f"Counted {count_value} records in {item.table}")
        return {
            "table": item.table,
            "count": count_value
        }
    except HTTPException as e:
        db.rollback()
        raise e
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"SQLAlchemy error during count: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    except Exception as e:
        db.rollback()
        logger.error(f"Unexpected error during count: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
    finally:
        if db:
            db.close()
