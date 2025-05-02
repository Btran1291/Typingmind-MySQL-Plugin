# TypingMind MySQL Plugin Server 🚀

This repository contains the backend server code for TypingMind plugins that enable interaction with MySQL databases. The server is built using FastAPI and SQLAlchemy, providing a dynamic and secure way to query and modify data in any MySQL database by receiving credentials directly in each request.

---

## Associated TypingMind Plugins

This server is designed to work with the following TypingMind plugins. You will need to install these plugins in your TypingMind application:

1. **MySQL Schema**  
   [https://cloud.typingmind.com/plugins/p-01JT8R2C7YC00QXN0Q0PF8SQ6F](https://cloud.typingmind.com/plugins/p-01JT8R2C7YC00QXN0Q0PF8SQ6F)

2. **MySQL Query**  
   [https://cloud.typingmind.com/plugins/p-01JT8R2TV312878TKY5A0MNYYQ](https://cloud.typingmind.com/plugins/p-01JT8R2TV312878TKY5A0MNYYQ)

3. **MySQL CRUD Operation**  
   [https://cloud.typingmind.com/plugins/p-01JT8R37PFQ7C3562X51E7JBCV](https://cloud.typingmind.com/plugins/p-01JT8R37PFQ7C3562X51E7JBCV)

---

## Deployment Options

You can deploy this server either on a cloud platform like Render.com or locally on your own computer.

### Option 1: Deploy to Render.com (Cloud Deployment)

1. **Create a New Web Service on Render:**  
   - Go to [Render Dashboard](https://dashboard.render.com/) and log in.  
   - Create a new Web Service and connect to this GitHub repository `Btran1291/Typingmind-MySQL-Plugin`.

2. **Configure Your Web Service:**  
   - **Name:** Choose a descriptive name (e.g., `mysql-plugin-server`).  
   - **Environment:** Select **Python**.  
   - **Branch:** Select the branch to deploy (usually `main`).  
   - **Build Command:** `pip install -r requirements.txt`  
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`  
   - **Instance Type:** Choose based on your needs (e.g., Free tier available).  
   - **Environment Variables:** No database credentials needed here; they are provided per request by TypingMind plugin user settings.

3. **Create and Deploy:**  
   - Click **Create Web Service** and wait for deployment to complete.

4. **Access Your Server:**  
   - Use the provided Render URL in your TypingMind plugin settings.

---

### Option 2: Deploy Locally (On Your Computer)

1. **Prerequisites:**  
   - Python 3.7+ installed.  
   - MySQL server running locally or accessible remotely.  
   - Command line interface (Terminal, PowerShell, etc.).

2. **Clone the Repository:**  
   ```bash
   git clone https://github.com/Btran1291/Typingmind-MySQL-Plugin.git
   cd Typingmind-MySQL-Plugin
   ```

3. **Create and Activate Virtual Environment:**  
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

4. **Install Dependencies:**  
   ```bash
   pip install -r requirements.txt
   ```

5. **Start the Server:**  
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

6. **Access the Server:**  
   - Open `http://localhost:8000` in your browser.  
   - API docs available at `http://localhost:8000/docs`.

---

## User Settings Input Guide

### Required Settings for MySQL Connection

- **Plugin Server URL:** The URL where this backend server is hosted (e.g., `https://your-server.com` or `http://localhost:8000`).

- **MySQL Host:** Hostname or IP of your MySQL server (e.g., `localhost`, `db.example.com`).

- **MySQL Database:** Name of the MySQL database to connect to.

- **MySQL Username:** Username with appropriate permissions.

- **MySQL Password:** Password for the MySQL user.

### Tips for Common MySQL Servers

- **AWS Aurora:**  
  - Use the **writer endpoint** for write operations; reader endpoint for read-only queries.  
  - Configure **VPC security groups** to allow connections from the plugin server's IP.  
  - Consider using **IAM authentication** for enhanced security.

- **Other Cloud Providers:**  
  - Ensure firewall rules permit your API server to connect.  
  - Use secure credentials and managed identity services if available.

- **Local MySQL:**  
  - Ensure MySQL is running and accessible via TCP.  
  - Use a user with least privilege necessary.

### Security Considerations

- Limit user privileges.  
- Use strong passwords and rotate them regularly.  
- Use encrypted connections (TLS/SSL) where possible.  
- Restrict API access via authentication (e.g., API keys).

---

## Plugin Documentation

### MySQL Schema Plugin

**Overview:**  
Allows you to connect to your MySQL database and retrieve detailed information about its structure, including a list of all tables and optionally their columns with metadata.

**Use Cases:**  
- Discover tables in your database.  
- Understand the structure of specific tables.  
- Plan accurate queries using schema information.

**Example AI Queries:**  
- "List all tables in my MySQL database."  
- "What columns does the `customers` table have?"  
- "Show me the structure of the `orders` table."

---

### MySQL Query Plugin

**Overview:**  
Enables flexible, dynamic read-only queries on your MySQL database, supporting filtering, joins, sorting, pagination, aggregation, and grouping.

**Use Cases:**  
- Retrieve specific data with complex filters.  
- Perform analytical queries with aggregation and grouping.  
- Join related tables to enrich query results.

**Example AI Queries:**  
- "Show me all orders placed in the last month."  
- "List customers who have placed more than 5 orders."  
- "Get the total sales per product category."  
- "Find all products supplied by 'Acme Corp' with price over $50."

---

### MySQL CRUD Operation Plugin

**Overview:**  
Provides full Create, Read, Update, and Delete capabilities, including batch operations, allowing comprehensive data manipulation in your MySQL database.

**Use Cases:**  
- Insert new records or bulk import data.  
- Update existing records individually or in batches.  
- Delete records safely using filters.  
- Perform complex data management tasks programmatically.

**Example AI Queries:**  
- "Add a new customer with name 'John Doe' and email 'john@example.com'."  
- "Update the price of 'Widget A' to $30."  
- "Delete all orders with status 'Cancelled'."  
- "Batch update the status of orders placed before 2023 to 'Archived'."

---

## Support & Contributions

Feel free to open issues or pull requests on [GitHub](https://github.com/Btran1291/Typingmind-MySQL-Plugin) for support or to contribute improvements.
