import logging
import traceback # Import traceback for logging full exceptions
from fastapi import FastAPI, Depends, HTTPException, Body
from sqlalchemy import create_engine, MetaData, Table, select, and_, or_, text, func
from sqlalchemy.orm import sessionmaker, Session, aliased
from sqlalchemy.sql import operators
from sqlalchemy.exc import IntegrityError, SQLAlchemyError, OperationalError, ProgrammingError # Import specific DB errors
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

# --- Connection Pooling Cache ---
_engine_cache = {}
_engine_cache_lock = Lock()

def get_engine_and_session_factory(credentials: DatabaseCredentials):
    """
    Given DatabaseCredentials, return a tuple of (engine, metadata, SessionLocal).
    Caches engines to avoid creating multiple engines for same credentials.
    """
    connection_url = (
        f"mysql+mysqlconnector://{credentials.mysql_user}:"
        f"{credentials.mysql_password}@{credentials.mysql_host}/"
        f"{credentials.mysql_database}"
    )
    with _engine_cache_lock:
        if connection_url in _engine_cache:
            engine, metadata, SessionLocal = _engine_cache[connection_url]
            logger.debug(f"Using cached engine for {credentials.mysql_user}@{credentials.mysql_host}/{credentials.mysql_database}")
        else:
            try:
                engine = create_engine(
                    connection_url,
                    pool_pre_ping=True,
                    pool_size=5,
                    max_overflow=10,
                    pool_recycle=3600,
                    connect_args={"connect_timeout": 10} # Add connection timeout
                )
                metadata = MetaData()
                metadata.reflect(bind=engine)
                SessionLocal = sessionmaker(bind=engine)
                _engine_cache[connection_url] = (engine, metadata, SessionLocal)
                logger.info(f"Created and cached new engine for {credentials.mysql_user}@{credentials.mysql_host}/{credentials.mysql_database}")
            except Exception as e:
                logger.error(f"Failed to create or reflect database engine: {e}", exc_info=True)
                # Provide a more specific error for connection failures
                error_detail = str(e)
                if "Access denied" in error_detail:
                     raise HTTPException(status_code=401, detail="Database authentication failed. Check credentials.")
                elif "Unknown database" in error_detail:
                     raise HTTPException(status_code=400, detail="Database not found. Check database name.")
                elif "Can't connect to MySQL server" in error_detail or "timed out" in error_detail:
                     raise HTTPException(status_code=504, detail="Cannot connect to database server. Check host and network.")
                else:
                     raise HTTPException(status_code=500, detail=f"Database connection error: {error_detail}")

    return engine, metadata, SessionLocal

# --- Validation Helpers ---
def validate_table_exists(table_name: str, metadata: MetaData):
    if table_name not in metadata.tables:
        raise HTTPException(status_code=400, detail=f"Table '{table_name}' does not exist.")

def validate_columns_exist(
    columns: List[str],
    alias_map: Dict[str, Table],
    main_table: Table,
    context: str = "columns"
):
    """
    Validate that each column in the list exists by attempting to get its object.
    `context` is a string used in error messages.
    """
    for col_name in columns:
        try:
            # Attempt to get the column object. get_column_from_alias_map raises HTTPException if not found or invalid format.
            get_column_from_alias_map(col_name, alias_map, main_table)
        except HTTPException as e:
            # Re-raise the HTTPException with context added to the detail
            raise HTTPException(status_code=e.status_code, detail=f"Column '{col_name}' in {context} validation failed: {e.detail}")
        except Exception as e:
             # Catch any other unexpected errors during lookup
             raise HTTPException(status_code=500, detail=f"Unexpected error validating column '{col_name}' in {context}: {e}")

def validate_filters_exist(filters: Optional[Filters], alias_map: Dict[str, Table], main_table: Table):
    """
    Recursively validate that all columns used in filters exist by attempting to get their objects.
    """
    if not filters:
        return
    for cond in filters.conditions:
        if isinstance(cond, Filters):
            validate_filters_exist(cond, alias_map, main_table)
        elif isinstance(cond, FilterCondition):
            # Reuse validate_columns_exist for filter columns
            validate_columns_exist([cond.column], alias_map, main_table, context="filter")

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


# --- Helper Functions ---
def get_table(table_name: str, metadata: MetaData) -> Table:
    if table_name not in metadata.tables:
        raise HTTPException(status_code=400, detail=f"Table '{table_name}' does not exist.")
    return metadata.tables[table_name]

def get_aliased_table(table_name: str, alias: Optional[str], alias_map: Dict[str, Table], metadata: MetaData) -> Table:
    if alias:
        if alias in alias_map:
            return alias_map[alias]
        base_table = get_table(table_name, metadata)
        aliased_table = aliased(base_table, name=alias)
        alias_map[alias] = aliased_table
        return aliased_table
    else:
        return get_table(table_name, metadata)

def get_column_from_alias_map(col_name: str, alias_map: Dict[str, Table], main_table: Table):
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
            raise HTTPException(status_code=400, detail=f"Alias or table '{alias_or_table}' not found.")
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

def build_filter_condition_with_alias_map(condition: FilterCondition, alias_map: Dict[str, Table], main_table: Table):
    col = get_column_from_alias_map(condition.column, alias_map, main_table)
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

def build_filters_with_alias_map(filters: Filters, alias_map: Dict[str, Table], main_table: Table):
    if not filters or not filters.conditions:
        return text("1=1")
    conditions = []
    for cond in filters.conditions:
        if isinstance(cond, Filters):
            conditions.append(build_filters_with_alias_map(cond, alias_map, main_table))
        elif isinstance(cond, FilterCondition):
            conditions.append(build_filter_condition_with_alias_map(cond, alias_map, main_table))
        else:
            raise HTTPException(status_code=400, detail=f"Invalid filter condition type: {type(cond)}")
    if filters.logic == "and":
        return and_(*conditions)
    elif filters.logic == "or":
        return or_(*conditions)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported logic operator: '{filters.logic}'.")

def apply_joins(stmt, main_table, joins: List[Join], alias_map: Dict[str, Table], metadata: MetaData):
    for join in joins:
        join_table = get_aliased_table(join.table, join.alias, alias_map, metadata)
        on_clauses = []
        for left_col, right_col in join.on.items():
            left_column = get_column_from_alias_map(left_col, alias_map, main_table)
            right_column = get_column_from_alias_map(right_col, alias_map, main_table)
            on_clauses.append(left_column == right_column)
        on_condition = and_(*on_clauses)
        join_type = join.type or "inner"
        if join_type == "inner":
            stmt = stmt.join(join_table, on_condition)
        elif join_type == "left":
            stmt = stmt.outerjoin(join_table, on_condition)
        # TODO: Add support for right and full outer joins if needed
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported join type: {join_type}")
        if join.columns:
            # Validate columns in join.columns
            validate_columns_exist(join.columns, alias_map, join_table, context=f"join '{join.alias or join.table}' columns")
            cols = [join_table.c[col] for col in join.columns]
            stmt = stmt.add_columns(*cols)
        if join.filters:
            # Validate filters in join.filters
            validate_filters_exist(join.filters, alias_map, join_table)
            filter_condition = build_filters_with_alias_map(join.filters, alias_map, join_table)
            stmt = stmt.where(filter_condition)
    return stmt

def apply_order_by(stmt, order_by_list: Optional[List[OrderBy]], alias_map: Dict[str, Table], main_table: Table):
    if not order_by_list:
        return stmt
    order_clauses = []
    for order in order_by_list:
        # Validation for order by columns is done before calling this function
        col = get_column_from_alias_map(order.column, alias_map, main_table)
        if order.direction.lower() == "asc":
            order_clauses.append(col.asc())
        elif order.direction.lower() == "desc":
            order_clauses.append(col.desc())
        else:
            raise HTTPException(status_code=400, detail=f"Invalid order direction: {order.direction}")
    return stmt.order_by(*order_clauses)

def build_aggregation_clauses(aggregations: List[Aggregation], alias_map: Dict[str, Table], main_table: Table):
    agg_clauses = []
    for agg in aggregations:
        if agg.function.lower() == "count" and agg.column is None:
            col = func.count()
        elif agg.column:
            # Validation for aggregation columns is done before calling this function
            col = get_column_from_alias_map(agg.column, alias_map, main_table)
            agg_func = getattr(func, agg.function.lower(), None)
            if agg_func is None:
                raise HTTPException(status_code=400, detail=f"Unsupported aggregation function: '{agg.function}'.")
            col = agg_func(col)
        else:
            raise HTTPException(status_code=400, detail=f"Column is required for aggregation function '{agg.function}'.")
        agg_clauses.append(col.label(agg.alias))
    return agg_clauses

def build_group_by_clauses(group_by_cols: List[str], alias_map: Dict[str, Table], main_table: Table):
    group_by_clauses = []
    for col_name in group_by_cols:
        # Validation for group by columns is done before calling this function
        col = get_column_from_alias_map(col_name, alias_map, main_table)
        group_by_clauses.append(col)
    return group_by_clauses

@app.get("/")
async def root():
    return {"message": "TypingMind MySQL API is running"}

@app.post("/schema")
async def get_schema(credentials: DatabaseCredentials):
    engine, metadata, _ = get_engine_and_session_factory(credentials)
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
    engine, metadata, SessionLocal = get_engine_and_session_factory(query.credentials)
    db = SessionLocal()
    try:
        # --- Explicit Input Validation ---
        validate_table_exists(query.table, metadata)
        main_table = get_table(query.table, metadata) # Get table object after validation

        alias_map = {main_table.name: main_table}
        if query.joins:
            # Populate alias_map early and validate join tables
            for join in query.joins:
                validate_table_exists(join.table, metadata) # Validate joined table exists
                get_aliased_table(join.table, join.alias, alias_map, metadata) # Populate alias_map

        # Validate columns in 'columns' field if present
        if query.columns:
            validate_columns_exist(query.columns, alias_map, main_table, context="columns")

        # Validate filters
        validate_filters_exist(query.filters, alias_map, main_table)

        # Validate columns in group_by if present
        if query.group_by:
            validate_columns_exist(query.group_by, alias_map, main_table, context="group_by")

        # Validate columns in order_by if present
        if query.order_by:
            order_by_cols = [order.column for order in query.order_by]
            validate_columns_exist(order_by_cols, alias_map, main_table, context="order_by")

        # Validate columns in aggregations if present
        if query.aggregations:
            agg_cols = [agg.column for agg in query.aggregations if agg.column]
            if agg_cols:
                validate_columns_exist(agg_cols, alias_map, main_table, context="aggregations")

        # --- Build select_items based on Aggregation/Grouping or Standard Query ---
        select_items = []
        is_aggregating = query.aggregations is not None and len(query.aggregations) > 0
        is_grouping = query.group_by is not None and len(query.group_by) > 0

        if is_aggregating or is_grouping:
            if is_grouping:
                group_by_clauses = build_group_by_clauses(query.group_by, alias_map, main_table)
                select_items.extend(group_by_clauses)
            if is_aggregating:
                agg_clauses = build_aggregation_clauses(query.aggregations, alias_map, main_table)
                select_items.extend(agg_clauses)
            if query.columns is not None and len(query.columns) > 0:
                raise HTTPException(status_code=400, detail="The 'columns' field cannot be used when 'aggregations' or 'group_by' are specified. Selected columns are defined by 'group_by' and 'aggregations'.")
        else:
            if query.columns:
                select_items = [get_column_from_alias_map(col, alias_map, main_table) for col in query.columns]
            else:
                select_items = [main_table]

        # --- Build the select statement ---
        stmt = select(*select_items).select_from(main_table)

        # --- Apply Joins ---
        if query.joins:
            stmt = apply_joins(stmt, main_table, query.joins, alias_map, metadata)

        # --- Apply Filters ---
        if query.filters:
            stmt = stmt.where(build_filters_with_alias_map(query.filters, alias_map, main_table))

        # --- Apply Group By ---
        if is_grouping:
             stmt = stmt.group_by(*group_by_clauses)


        # --- Apply Sorting ---
        if query.order_by:
            stmt = apply_order_by(stmt, query.order_by, alias_map, main_table)


        # --- Apply Pagination ---
        if query.limit is not None:
            stmt = stmt.limit(query.limit)
        if query.offset:
            stmt = stmt.offset(query.offset)


        # --- DEBUGGING PRINT STATEMENTS (Optional for Production) ---
        # print("\n--- Generated SQL Statement ---")
        # print(stmt)
        # print("-----------------------------\n")
        # --- END DEBUGGING ---

        # Execute the query
        result = db.execute(stmt)

        # --- DEBUGGING PRINT STATEMENTS (Optional for Production) ---
        # print("\n--- Raw Query Result ---")
        # raw_rows = result.fetchall()
        # print(raw_rows)
        # print("------------------------\n")
        # --- END DEBUGGING ---

        # Fetch results and convert to list of dictionaries
        # Use result.keys() to get column names for dictionary keys
        # If using raw_rows from debugging, uncomment and use: rows = [dict(zip(result.keys(), row)) for row in raw_rows]
        rows = [dict(zip(result.keys(), row)) for row in result]


        return {"data": rows}

    except HTTPException as e:
        # Re-raise HTTP exceptions for FastAPI to handle
        raise e
    except Exception as e:
        # Catch any other unexpected errors during query building or execution
        logger.error(f"Unexpected error during query execution: {e}", exc_info=True)
        # Provide a more user-friendly error message
        detail = str(e)
        # Catch common MySQL error for non-aggregated columns in select without group by
        if "1055" in detail or "GROUP BY" in detail:
             detail = "Query error: SELECT list contains nonaggregated column which is not included in GROUP BY clause."
        # Catch common MySQL error for unknown column
        elif "Unknown column" in detail:
             detail = f"Query error: {detail}" # Keep the specific column name from the DB error
        # Catch common MySQL syntax error
        elif "syntax error" in detail:
             detail = f"Query error: {detail}" # Keep the specific syntax error
        else:
             detail = "An internal server error occurred." # Generic error for others

        raise HTTPException(status_code=500, detail=detail)
    finally:
        # Ensure the session is closed
        if db:
            db.close()


@app.post("/insert")
async def insert_data(item: Insert):
    engine, metadata, SessionLocal = get_engine_and_session_factory(item.credentials)
    db = SessionLocal()
    try:
        # --- Explicit Input Validation ---
        validate_table_exists(item.table, metadata)
        table = get_table(item.table, metadata) # Get table object after validation
        validate_insert_data_columns(item.data, table) # Validate columns in data

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
    engine, metadata, SessionLocal = get_engine_and_session_factory(item.credentials)
    db = SessionLocal()
    try:
        # --- Explicit Input Validation ---
        validate_table_exists(item.table, metadata)
        table = get_table(item.table, metadata) # Get table object after validation
        # Validate columns in data for each item in the batch (assuming consistent schema)
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
    engine, metadata, SessionLocal = get_engine_and_session_factory(item.credentials)
    db = SessionLocal()
    try:
        # --- Explicit Input Validation ---
        validate_table_exists(item.table, metadata)
        table = get_table(item.table, metadata) # Get table object after validation
        validate_update_data_columns(item.data, table) # Validate columns in data
        # Validate filters (uses alias_map, but for update/delete it's empty)
        validate_filters_exist(item.filters, {}, table)


        stmt = table.update()
        if item.filters:
            filter_condition = build_filters_with_alias_map(item.filters, {}, table)
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
    engine, metadata, SessionLocal = get_engine_and_session_factory(item.credentials)
    db = SessionLocal()
    total_rows_affected = 0
    try:
        # --- Explicit Input Validation ---
        validate_table_exists(item.table, metadata)
        table = get_table(item.table, metadata) # Get table object after validation
        if not item.updates:
            logger.info(f"No update specifications provided for table '{item.table}'.")
            return {
                "message": f"No update specifications provided for table '{item.table}'.",
                "total_rows_affected": 0
            }
        # Validate columns and filters for each update item
        for update_item in item.updates:
             validate_update_data_columns(update_item.data, table)
             validate_filters_exist(update_item.filters, {}, table)


        for update_item in item.updates:
            stmt = table.update()
            filter_condition = build_filters_with_alias_map(update_item.filters, {}, table)
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
    engine, metadata, SessionLocal = get_engine_and_session_factory(item.credentials)
    db = SessionLocal()
    try:
        # --- Explicit Input Validation ---
        validate_table_exists(item.table, metadata)
        table = get_table(item.table, metadata) # Get table object after validation
        # Validate filters (filters are required by Pydantic model)
        validate_filters_exist(item.filters, {}, table)


        stmt = table.delete()
        filter_condition = build_filters_with_alias_map(item.filters, {}, table)
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
    engine, metadata, SessionLocal = get_engine_and_session_factory(item.credentials)
    db = SessionLocal()
    total_rows_affected = 0
    try:
        # --- Explicit Input Validation ---
        validate_table_exists(item.table, metadata)
        table = get_table(item.table, metadata) # Get table object after validation
        if not item.deletions:
            logger.info(f"No delete specifications provided for table '{item.table}'.")
            return {
                "message": f"No delete specifications provided for table '{item.table}'.",
                "total_rows_affected": 0
            }
        # Validate filters for each delete item
        for delete_item in item.deletions:
             validate_filters_exist(delete_item.filters, {}, table)


        for delete_item in item.deletions:
            stmt = table.delete()
            filter_condition = build_filters_with_alias_map(delete_item.filters, {}, table)
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
    engine, metadata, SessionLocal = get_engine_and_session_factory(item.credentials)
    db = SessionLocal()
    try:
        # --- Explicit Input Validation ---
        validate_table_exists(item.table, metadata)
        table = get_table(item.table, metadata) # Get table object after validation
        # Validate filters
        validate_filters_exist(item.filters, {}, table)


        stmt = select(func.count()).select_from(table)
        if item.filters:
            filter_condition = build_filters_with_alias_map(item.filters, {}, table)
            stmt = stmt.where(filter_condition)

        result = db.execute(stmt)
        count_value = result.scalar()
        logger.info(f"Counted {count_value} records in {item.table}")
        return {
            "table": item.table,
            "count": count_value
        }
    except HTTPException as e:
        db.rollback() # Rollback is not strictly needed for reads, but harmless
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
