# TypingMind MySQL Plugin Server 🚀

This repository contains the backend server code for TypingMind plugins that enable interaction with MySQL databases. The server is built using FastAPI and SQLAlchemy, providing a dynamic and secure way to query and modify data in any MySQL database by receiving credentials directly in each request.

---

## Associated TypingMind Plugins

This server is designed to work with the following TypingMind plugins. You will need to install these plugins in your TypingMind application:

1. [**MySQL Schema**](https://cloud.typingmind.com/plugins/p-01JZPTRSR34C5WZ49KT1VKSSB0)  

2. [**MySQL Query**](https://cloud.typingmind.com/plugins/p-01K02NK8QMCTTAS3GJ2ZXGFCT2)  

3. [**MySQL CRUD Operation**](https://cloud.typingmind.com/plugins/p-01JZPTSXS1JZD0J5YPNRKVR8PV)  

4. [**MySQL DDL Operation**](https://cloud.typingmind.com/plugins/p-01JZPTNSGK71TCYNTGFX27Q5ZS)
---

## Deployment Options

You can deploy this server either on a cloud platform like Render.com or locally on your own computer.

### Option 1: Deploy to Render.com (Cloud Deployment)

1.  **Create a New Web Service on Render:**
    *   Go to [Render Dashboard](https://dashboard.render.com/) and log in.
    *   Create a new Web Service and connect to this GitHub repository `Btran1291/Typingmind-MySQL-Plugin`.

2.  **Configure Your Web Service:**
    *   **Name:** Choose a descriptive name (e.g., `mysql-plugin-server`).
    *   **Environment:** Select **Python**.
    *   **Branch:** Select `main`.
    *   **Build Command:** `pip install -r requirements.txt`
    *   **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
    *   **Instance Type:** Choose based on your needs (e.g., Free tier).

3.  **Create and Deploy:**
    *   Click **Create Web Service** and wait for deployment to complete.

4.  **Access Your Server:**
    *   Use the provided Render URL in your TypingMind plugin settings.

---

### Option 2: Deploy Locally (On Your Computer)

1.  **Prerequisites:**
    *   Python 3.7+ installed.
    *   MySQL server running locally or accessible remotely.
    *   Command line interface (Terminal, PowerShell, etc.).

2.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Btran1291/Typingmind-MySQL-Plugin.git
    cd Typingmind-MySQL-Plugin
    ```

3.  **Create and Activate Virtual Environment:**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Start the Server:**
    ```bash
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
    ```

6.  **Access the Server:**
    *   Open `http://localhost:8000` in your browser.
    *   API docs available at `http://localhost:8000/docs`.

---

## User Settings Input Guide

### Required Settings for MySQL Connection

-   **Plugin Server URL:** The URL where this backend server is hosted (e.g., `https://your-server.render.com` or `http://localhost:8000`).

-   **MySQL Host:** Hostname or IP of your MySQL server (e.g., `localhost`, `db.example.com`).

-   **MySQL Database:** Name of the MySQL database to connect to.

-   **MySQL Username:** Username with appropriate permissions.

-   **MySQL Password:** Password for the MySQL user.

---

### **Tips for Common MySQL Servers**

To connect successfully, you often need to configure your MySQL server to accept remote connections from your plugin server. Here are tips for common hosting environments:

#### **For Local MySQL (XAMPP, WAMP, MAMP, Docker)**
-   **Hostname:** When connecting from a cloud-deployed server, you **must** use a tunneling service like **Ngrok** or **Cloudflare Tunnel** (see instructions below). The hostname will be the public URL provided by the tunnel service (e.g., `X.tcp.ngrok.io:YYYYY`).
-   **`bind-address`:** By default, MySQL only listens for connections from `localhost` (`127.0.0.1`). To allow connections through a tunnel, you must edit your MySQL configuration file (`my.ini` in XAMPP/WAMP, `my.cnf` in MAMP/Linux) and change `bind-address = 127.0.0.1` to `bind-address = 0.0.0.0`. **Remember to restart the MySQL server after this change.**
-   **Firewall:** Your local computer's firewall (e.g., Windows Defender Firewall) must allow inbound connections on your MySQL port (typically `3306`).

#### **For AWS RDS & Aurora**
-   **Hostname:** Use the **Endpoint** URL provided in your RDS or Aurora cluster dashboard (e.g., `your-db-instance.random-chars.us-east-1.rds.amazonaws.com`).
-   **Security Groups:** This is the most common issue. You must edit the database's **VPC Security Group** and add an **Inbound Rule** to allow traffic on port `3306` from your plugin server's IP address.
    -   If your Render service is on a paid plan with a **Static Outbound IP**, add that IP address to the rule (e.g., `YOUR_RENDER_IP/32`).
    -   If you are on a free plan with dynamic IPs, you may need to allow all traffic (`0.0.0.0/0`), but this is **highly insecure** for production.

#### **For Google Cloud SQL & Azure Database for MySQL**
-   **Hostname:** Use the **Public IP address** or **Connection Name** provided in the Google Cloud or Azure portal.
-   **Firewall Rules:**
    -   In **Google Cloud SQL**, go to "Connections" -> "Networking" and add your Render server's IP address to the **Authorized networks** list.
    -   In **Azure**, go to your database's "Connection security" settings and add a **Firewall rule** to allow your Render server's IP address.
    -   The same advice about static vs. dynamic IPs from the AWS section applies here.

#### **For DigitalOcean Managed Databases**
-   **Hostname:** Use the **Host** provided in your DigitalOcean database cluster's "Connection Details".
-   **Trusted Sources:** This is DigitalOcean's term for IP whitelisting. You must edit the **Trusted Sources** for your database cluster and add the IP address of your Render server.

#### **For Hosting with cPanel / Plesk**
-   **Hostname:** This is often your domain name (e.g., `yourdomain.com`) or a specific database hostname provided by your hosting provider. It is almost never `localhost`.
-   **Remote MySQL:** You must log in to your cPanel/Plesk dashboard and find the **"Remote MySQL"** feature. In this section, you need to add the IP address of your Render server to the list of **Access Hosts**. Using the wildcard `%` is possible but less secure.

#### **For DIY MySQL on a VPS (DigitalOcean, Linode, etc.)**
-   **Hostname:** The public IP address of your VPS.
-   **`bind-address`:** Similar to the local setup, you must edit your MySQL configuration file (`/etc/mysql/my.cnf` or similar) and set `bind-address = 0.0.0.0` to allow connections from outside the VPS.
-   **Firewall (UFW):** You must configure the server's firewall to allow incoming connections on port 3306. For UFW on Ubuntu, the command is `sudo ufw allow 3306/tcp`.

---

### **Connecting to Local MySQL Databases from Cloud Deployment (e.g., Render.com)**

**Important Note:** If your FastAPI server is deployed on a cloud service like Render.com, it **cannot directly access** a MySQL database hosted on your local computer or a client's local network. This is because `localhost` or private IP addresses (like `192.168.x.x`) refer to the cloud server itself, not your local machine.

To bridge this gap and allow your cloud-deployed server to connect to a local MySQL database, you need to use a **secure tunneling service**. These services create a public endpoint that forwards traffic to your local machine.

#### **Option A: Using Ngrok**

Ngrok creates a temporary, public URL that tunnels traffic to a port on your local machine.

1.  **On the Local Machine (where MySQL is running):**
    *   **Download Ngrok:** Get the executable from [ngrok.com/download](https://ngrok.com/download) and unzip it.
    *   **Start Ngrok Tunnel:** Open a terminal/command prompt in the Ngrok directory and run:
        ```bash
        ./ngrok tcp 3306
        ```
        (Replace `3306` with your MySQL server's actual port if different).
    *   **Copy Forwarding Address:** Ngrok will display a `Forwarding` address like `tcp://X.tcp.ngrok.io:YYYYY`. Copy the entire `X.tcp.ngrok.io:YYYYY` part.
    *   **Keep Ngrok Running:** The terminal window running Ngrok **must remain open** for the tunnel to stay active.
    *   **Firewall:** Ensure your local machine's firewall allows inbound connections on port `3306` (or your MySQL port).

2.  **In TypingMind Plugin Settings:**
    *   **MySQL Host:** Paste the **entire Ngrok forwarding address** you copied (e.g., `X.tcp.ngrok.io:YYYYY`).
    *   **MySQL Database, MySQL Username, MySQL Password:** Use your local MySQL database's actual credentials.

**Caveats:** Ngrok's free tier provides dynamic URLs that change each time the tunnel is restarted. This is suitable for testing but not for stable production use.

#### **Option B: Using Cloudflare Tunnel (for more stable & secure access)**

Cloudflare Tunnel (`cloudflared`) creates a secure, persistent, outbound-only connection to Cloudflare's network, exposing your local service via a stable hostname.

1.  **On the Local Machine (where MySQL is running):**
    *   **Download `cloudflared`:** Get the executable from [developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation](https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation).
    *   **Log in to Cloudflare:** Run `cloudflared login` in your terminal and follow the browser prompts (requires a free Cloudflare account).
    *   **Create a Tunnel:** Run `cloudflared tunnel create <YOUR_TUNNEL_NAME>`.
    *   **Configure DNS (Optional but Recommended):** Route a subdomain to your tunnel (e.g., `db.yourdomain.com`) via your Cloudflare DNS settings, or use the public hostname Cloudflare provides.
    *   **Create Configuration File:** Create a `config.yml` file (e.g., in `~/.cloudflared/`) with your tunnel ID and the service to expose:
        ```yaml
        tunnel: <YOUR_TUNNEL_ID>
        credentials-file: /path/to/your/credentials.json # Path provided by cloudflared create
        ingress:
          - hostname: db.yourdomain.com # Or the public hostname Cloudflare gave you
            service: tcp://localhost:3306 # Your local MySQL port
          - service: http_status:404
        ```
    *   **Run the Tunnel:** Run `cloudflared tunnel run <YOUR_TUNNEL_NAME>` (or configure it as a system service for persistence).
    *   **Firewall:** Ensure your local machine's firewall allows inbound connections on port `3306` (or your MySQL port).

2.  **In TypingMind Plugin Settings:**
    *   **MySQL Host:** Use the **stable hostname** configured in Cloudflare (e.g., `db.yourdomain.com` or the Cloudflare-provided public hostname).
    *   **MySQL Database, MySQL Username, MySQL Password:** Use your local MySQL database's actual credentials.

**Benefits:** Cloudflare Tunnel offers stable URLs, enhanced security (outbound connections only from the client's machine, no inbound firewall holes needed on the router), DDoS protection, and can be run as a persistent service.

---

## Plugin Documentation

### MySQL Schema Plugin

**Overview:**
Allows you to connect to your MySQL database and retrieve detailed information about its structure, including a list of all tables and optionally their columns with metadata.

**Use Cases:**
-   Discover tables in your database.
-   Understand the structure of specific tables.
-   Plan accurate queries using schema information.

**Example AI Queries:**
-   "List all tables in my MySQL database."
-   "What columns does the `customers` table have?"
-   "Show me the structure of the `orders` table."

---

### MySQL Query Plugin

**Overview:**
Enables flexible, dynamic read-only queries on your MySQL database, supporting filtering, joins, sorting, pagination, aggregation, and grouping.

**Use Cases:**
-   Retrieve specific data with complex filters.
-   Perform analytical queries with aggregation and grouping.
-   Join related tables to enrich query results.

**Example AI Queries:**
-   "Show me all orders placed in the last month."
-   "List customers who have placed more than 5 orders."
-   "Get the total sales per product category."
-   "Find all products supplied by 'Acme Corp' with price over $50."

---

### MySQL CRUD Operation Plugin

**Overview:**
Provides full Create, Read, Update, and Delete capabilities, including batch operations, allowing comprehensive data manipulation in your MySQL database.

**Use Cases:**
-   Insert new records or bulk import data.
-   Update existing records individually or in batches.
-   Delete records safely using filters.
-   Perform complex data management tasks programmatically.

**Example AI Queries:**
-   "Add a new customer with name 'John Doe' and email 'john@example.com'."
-   "Update the price of 'Widget A' to $30."
-   "Delete all orders with status 'Cancelled'."
-   "Batch update the status of orders placed before 2023 to 'Archived'."

---

### MySQL DDL Operation Plugin

**Overview:**  
This plugin empowers you to manage and modify the structure of your MySQL database. It provides functionalities to create, alter, and drop tables and columns, giving you direct control over your database schema.

**What it does:**  
- **Create Table:** Define and create new tables with specified columns and properties.  
- **Drop Table:** Remove existing tables from the database.  
- **Truncate Table:** Empty all records from a table while keeping its structure.  
- **Add Column:** Add new columns to an existing table.  
- **Drop Column:** Remove columns from an existing table.  
- **Rename Column:** Change the name of a column in a table.  
- **Modify Column:** Alter the definition (type, nullability, default, etc.) of an existing column.  
- **Rename Table:** Change the name of an existing table.

**Use Cases:**  
- **Setting up New Data Structures:** "Create a new table called 'products' with columns for 'id' (INT, primary key), 'name' (VARCHAR(255)), and 'price' (DECIMAL(10,2))."  
- **Adjusting Existing Schemas:** "Add a 'last_login' (DATETIME) column to the 'users' table." or "Change the 'description' column in 'products' to TEXT."  
- **Cleaning Up Old Structures:** "Drop the 'temp_logs' table."  
- **Refactoring Database Design:** "Rename the 'customer_id' column in the 'orders' table to 'user_id'." or "Rename the 'old_users' table to 'archived_users'."

---

## Security Considerations

- Limit user privileges to only necessary permissions for security.  
- Use strong, regularly rotated passwords.  
- Enable encrypted connections (TLS/SSL) where possible.  
- Restrict API access via authentication mechanisms such as API keys or IP whitelisting.

---

## Support & Contributions

Feel free to open issues or pull requests on [GitHub](https://github.com/Btran1291/Typingmind-MySQL-Plugin) for support or to contribute improvements.
