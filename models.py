from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field, validator

# Model for database connection credentials
class DatabaseCredentials(BaseModel):
    mysql_host: str = Field(..., description="The hostname or IP address of the MySQL server")
    mysql_database: str = Field(..., description="The name of the database to connect to")
    mysql_user: str = Field(..., description="The username for connecting to the database")
    mysql_password: str = Field(..., description="The password for the database user")


# Model for a single filter condition
class FilterCondition(BaseModel):
    column: str = Field(..., description="Column name, optionally prefixed with table or alias")
    operator: str = Field(..., description="Comparison operator", pattern="^(eq|neq|gt|gte|lt|lte|like|in|nin|is_null|is_not_null)$") # ADD is_null|is_not_null
    value: Union[str, int, float, List[Union[str, int, float]], None] = Field(..., description="Value or list of values for 'in'/'nin'. For 'is_null'/'is_not_null', this should be null.") # ADD None to Union, update description

# Model for compound filters (recursive)
class Filters(BaseModel):
    logic: Optional[str] = Field("and", description="Logical operator to combine conditions", pattern="^(and|or)$")
    conditions: List[Union['Filters', FilterCondition]] = Field(..., description="List of filter conditions or nested filters")

# Resolve forward references for recursive model
Filters.update_forward_refs()

# Model for join filters (same structure as main filters)
JoinFilters = Filters

# Model for a join specification
class Join(BaseModel):
    table: str = Field(..., description="Name of the table to join")
    alias: Optional[str] = Field(None, description="Alias for the joined table")
    on: Dict[str, str] = Field(..., description="Join condition mapping: main_table.column -> joined_table.column")
    type: Optional[str] = Field("inner", description="Join type", pattern="^(inner|left|right|full)$") # ADD 'right' and 'full' here
    columns: Optional[List[str]] = Field(None, description="Columns to select from joined table")
    filters: Optional[JoinFilters] = Field(None, description="Filters for the joined table")

# Model for order by instructions
class OrderBy(BaseModel):
    column: str = Field(..., description="Column name, optionally prefixed with table or alias")
    direction: str = Field(..., description="Sort direction", pattern="^(asc|desc)$")

# Model for aggregation specification
class Aggregation(BaseModel):
    function: str = Field(..., description="Aggregation function", pattern="^(count|sum|avg|min|max)$")
    column: Optional[str] = Field(None, description="Column name (optional for count(*))")
    alias: str = Field(..., description="Alias for the aggregated column in the result")
    distinct: Optional[bool] = Field(False, description="Whether to count only distinct values. Relevant for 'count' function.")

# Main query model
class Query(BaseModel):
    # Add credentials field
    credentials: DatabaseCredentials = Field(..., description="Database connection credentials")

    table: str = Field(..., description="Main table to query")
    columns: Optional[List[str]] = Field(None, description="Columns to select from main table")
    filters: Optional[Filters] = Field(None, description="Filters for main table")
    joins: Optional[List[Join]] = Field(None, description="Join specifications")
    order_by: Optional[List[OrderBy]] = Field(None, description="Sorting instructions")
    limit: Optional[int] = Field(None, ge=1, description="Limit number of records")
    offset: Optional[int] = Field(0, ge=0, description="Number of records to skip")
    aggregations: Optional[List[Aggregation]] = Field(None, description="List of aggregation specifications")
    group_by: Optional[List[str]] = Field(None, description="List of columns to group by")
    having: Optional[Filters] = Field(None, description="Filter conditions to apply to aggregated results (HAVING clause). Use with 'group_by'.")

    # Corrected validators
    @validator('group_by', always=True)
    def check_group_by_format(cls, v):
        return v

    @validator('aggregations', always=True)
    def check_aggregations_format(cls, v):
        return v

    @validator('columns', always=True)
    def check_columns_with_aggregations_and_group_by(cls, v, values):
        aggregations = values.get('aggregations')
        group_by = values.get('group_by')
        if (aggregations is not None and len(aggregations) > 0) or (group_by is not None and len(group_by) > 0):
             if v is not None and len(v) > 0:
                  raise ValueError("The 'columns' field cannot be used when 'aggregations' or 'group_by' are specified. Selected columns are defined by 'group_by' and 'aggregations'.")
        return v

# Model for insert operation
class Insert(BaseModel):
    # Add credentials field
    credentials: DatabaseCredentials = Field(..., description="Database connection credentials")

    table: str = Field(..., description="Name of the table to insert into")
    data: Dict[str, Any] = Field(..., description="The record data as a dictionary (column_name: value)")

# Model for update operation
class Update(BaseModel):
    # Add credentials field
    credentials: DatabaseCredentials = Field(..., description="Database connection credentials")

    table: str = Field(..., description="Name of the table to update")
    data: Dict[str, Any] = Field(..., description="Columns and their new values")
    filters: Optional[Filters] = Field(None, description="Filter conditions to select rows to update")

# Model for delete operation
class Delete(BaseModel):
    # Add credentials field
    credentials: DatabaseCredentials = Field(..., description="Database connection credentials")

    table: str = Field(..., description="Name of the table to delete from")
    filters: Filters = Field(..., description="Filter conditions to select rows to delete")

# Model for count operation
class Count(BaseModel):
    # Add credentials field
    credentials: DatabaseCredentials = Field(..., description="Database connection credentials")

    table: str = Field(..., description="Name of the table to count from")
    filters: Optional[Filters] = Field(None, description="Optional filter conditions to limit the count")

# Model for batch insert operation
class BatchInsert(BaseModel):
    # Add credentials field
    credentials: DatabaseCredentials = Field(..., description="Database connection credentials")

    table: str = Field(..., description="Name of the table to insert into")
    data: List[Dict[str, Any]] = Field(..., description="A list of record data dictionaries (column_name: value)")

# Model for batch update item
class BatchUpdateItem(BaseModel):
    data: Dict[str, Any] = Field(..., description="Columns and their new values for this update")
    filters: Filters = Field(..., description="Filter conditions to select rows for *this* update")

# Model for batch update operation
class BatchUpdate(BaseModel):
    # Add credentials field
    credentials: DatabaseCredentials = Field(..., description="Database connection credentials")

    table: str = Field(..., description="Name of the table to update")
    updates: List[BatchUpdateItem] = Field(..., description="A list of update specifications")

# Model for batch delete item
class BatchDeleteItem(BaseModel):
    filters: Filters = Field(..., description="Filter conditions to select rows for *this* deletion")

# Model for batch delete operation
class BatchDelete(BaseModel):
    # Add credentials field
    credentials: DatabaseCredentials = Field(..., description="Database connection credentials")

    table: str = Field(..., description="Name of the table to delete from")
    deletions: List[BatchDeleteItem] = Field(..., description="A list of delete specifications")

class SchemaRequest(BaseModel):
    mysql_host: str
    mysql_database: str
    mysql_user: str
    mysql_password: str
    include_columns: bool = True

class ColumnDefinition(BaseModel):
    name: str = Field(..., description="Name of the column")
    type: str = Field(..., description="SQL data type (e.g., 'VARCHAR(255)', 'INT', 'BOOLEAN', 'DATETIME')")
    nullable: Optional[bool] = Field(True, description="Whether the column can be NULL")
    primary_key: Optional[bool] = Field(False, description="Whether the column is part of the primary key")
    autoincrement: Optional[bool] = Field(False, description="Whether the column is auto-incrementing (e.g., for INT PRIMARY KEY)")
    default: Optional[Union[str, int, float, bool]] = Field(None, description="Default value for the column")
    unique: Optional[bool] = Field(False, description="Whether the column must contain unique values")

class CreateTable(BaseModel):
    credentials: DatabaseCredentials = Field(..., description="Database connection credentials")
    table_name: str = Field(..., description="Name of the new table to create")
    columns: List[ColumnDefinition] = Field(..., min_items=1, description="List of column definitions for the new table")
    if_not_exists: Optional[bool] = Field(False, description="If True, adds IF NOT EXISTS to the CREATE TABLE statement")

class DropTable(BaseModel):
    credentials: DatabaseCredentials = Field(..., description="Database connection credentials")
    table_name: str = Field(..., description="Name of the table to drop")
    if_exists: Optional[bool] = Field(False, description="If True, adds IF EXISTS to the DROP TABLE statement")

class TruncateTable(BaseModel):
    credentials: DatabaseCredentials = Field(..., description="Database connection credentials")
    table_name: str = Field(..., description="Name of the table to truncate")

class AddColumn(BaseModel):
    credentials: DatabaseCredentials = Field(..., description="Database connection credentials")
    table_name: str = Field(..., description="Name of the table to alter")
    column_definition: ColumnDefinition = Field(..., description="Definition of the new column to add")
    if_not_exists: Optional[bool] = Field(False, description="If True, adds IF NOT EXISTS to the ADD COLUMN statement (MySQL specific)")

class DropColumn(BaseModel):
    credentials: DatabaseCredentials = Field(..., description="Database connection credentials")
    table_name: str = Field(..., description="Name of the table to alter")
    column_name: str = Field(..., description="Name of the column to drop")
    if_exists: Optional[bool] = Field(False, description="If True, adds IF EXISTS to the DROP COLUMN statement (MySQL specific)")

class RenameColumn(BaseModel):
    credentials: DatabaseCredentials = Field(..., description="Database connection credentials")
    table_name: str = Field(..., description="Name of the table containing the column to rename")
    old_column_name: str = Field(..., description="Current name of the column")
    new_column_name: str = Field(..., description="New name for the column")

class ModifyColumn(BaseModel):
    credentials: DatabaseCredentials = Field(..., description="Database connection credentials")
    table_name: str = Field(..., description="Name of the table containing the column to modify")
    column_name: str = Field(..., description="Name of the column to modify")
    new_column_definition: ColumnDefinition = Field(..., description="The new definition for the column (type, nullability, default, etc.)")

class RenameTable(BaseModel):
    credentials: DatabaseCredentials = Field(..., description="Database connection credentials")
    old_table_name: str = Field(..., description="Current name of the table to rename")
    new_table_name: str = Field(..., description="New name for the table")
