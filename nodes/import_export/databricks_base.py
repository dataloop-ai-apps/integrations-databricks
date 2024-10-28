import dtlpy as dl
import logging
from databricks.sdk.core import oauth_service_principal, Config
from databricks import sql

logger = logging.getLogger(name='databricks-connect')


class DatabricksBase(dl.BaseServiceRunner):
    """
    A class for running a service that interacts with Databricks.
    """

    def __init__(self):
        """
        Initializes the ServiceRunner with Databricks credentials.
        """
        self.logger = logger

    def call_databricks_query(self, server_hostname: str, databricks_client_id: str, databricks_client_secret: str, databricks_http_path: str, query: str, params: tuple = None):
        """
        Executes a SQL query on Databricks using provided credentials and connection settings.

        :param server_hostname: The hostname of the Databricks server.
        :param databricks_client_id: The client ID for Databricks.
        :param databricks_client_secret: The client secret for Databricks.
        :param databricks_http_path: The HTTP path for Databricks.
        :param query: The SQL query to execute.
        :param params: Optional tuple of parameters for the SQL query.
        :return: The result of the SQL query.
        """
        self.logger.info(f"Executing query on server '{server_hostname}' with HTTP path '{databricks_http_path}'.")
        self.logger.debug(f"Query: {query}")
        if params:
            self.logger.debug(f"Query Parameters: {params}")

        def credential_provider():
            config = Config(
                host=f"https://{server_hostname}",
                client_id=databricks_client_id,
                client_secret=databricks_client_secret
            )
            return oauth_service_principal(config)

        try:
            with sql.connect(
                server_hostname=server_hostname,
                http_path=databricks_http_path,
                credentials_provider=credential_provider,
                connection_timeout=10
            ) as connection:
                cursor = connection.cursor()
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)

                # Fetch results for SELECT queries, otherwise commit changes
                if query.strip().upper().startswith("SELECT"):
                    result = cursor.fetchall()
                else:
                    connection.commit()
                    result = cursor.rowcount

                cursor.close()
                return result

        except Exception as e:
            self.logger.error(f"Failed to execute query: {e}")
            raise Exception(f"Failed to execute query: {e}")

    def create_table(self, server_hostname: str, databricks_client_id: str, databricks_client_secret: str, databricks_http_path: str, catalog: str, schema: str, table_name: str, dataset_id: str):
        """
        Retrieves data from Databricks and creates PromptItems based on the retrieved data.

        :param server_hostname: The hostname of the Databricks server.
        :param databricks_client_id: The client ID for Databricks.
        :param databricks_client_secret: The client secret for Databricks.
        :param databricks_http_path: The HTTP path for Databricks.
        :param catalog: The Databricks catalog.
        :param schema: The Databricks schema.
        :param table_name: The name of the table to query.
        :param dataset_id: The ID of the dataset to upload the results.
        :return: The uploaded items or None if an error occurs.
        """
        self.logger.info(f"Creating table for dataset '{dataset_id}' and table '{table_name}'.")

        try:
            dataset = dl.datasets.get(dataset_id=dataset_id)
            self.logger.info(f"Successfully retrieved dataset with ID '{dataset_id}'.")
        except Exception as e:
            self.logger.error(f"Failed to get dataset with ID '{dataset_id}': {e}")
            return None

        # Execute query to fetch data
        query = f"SELECT * FROM {catalog}.{schema}.{table_name}"
        result = self.call_databricks_query(server_hostname, databricks_client_id, databricks_client_secret, databricks_http_path, query)

        prompt_items = []
        for res in result:
            prompt_item = dl.PromptItem(name=str(res.id))
            prompt_item.add(
                message={
                    "role": "user",
                    "content": [{"mimetype": dl.PromptType.TEXT, "value": res.prompt}]
                }
            )
            prompt_items.append(prompt_item)

        # Upload PromptItems to Dataloop
        items = list(dataset.items.upload(local_path=prompt_items, overwrite=True))
        self.logger.info(f"Successfully uploaded {len(items)} items to dataset '{dataset_id}'.")
        return items

    def update_table(self, item: dl.Item, server_hostname: str, databricks_client_id: str, databricks_client_secret: str, databricks_http_path: str, catalog: str, schema: str, table_name: str):
        """
        Updates the specified table with best response information extracted from Dataloop item annotations.

        :param item: The Dataloop item to update in Databricks.
        :param server_hostname: The hostname of the Databricks server.
        :param databricks_client_id: The client ID for Databricks.
        :param databricks_client_secret: The client secret for Databricks.
        :param databricks_http_path: The HTTP path for Databricks.
        :param catalog: The Databricks catalog.
        :param schema: The Databricks schema.
        :param table_name: The Databricks table name.
        :return: The updated item or None if an error occurs.
        """
        self.logger.info(f"Updating table '{table_name}' for item with ID '{item.id}'.")

        prompt_item = dl.PromptItem.from_item(item)
        first_prompt_key = prompt_item.prompts[0].key

        # Find the best response based on annotation attributes
        best_response = None
        model_id, name = None, "human"  # Default value for 'name' if not found

        for resp in item.annotations.list():
            try:
                is_best = resp.attributes.get('isBest', False)
            except AttributeError:
                is_best = False
            if is_best and resp.metadata['system'].get('promptId') == first_prompt_key:
                best_response = resp.coordinates
                model_info = resp.metadata.get('user', {}).get('model', {})
                model_id = model_info.get('model_id', '')
                name = model_info.get('name', 'human')
                break

        if best_response is None:
            self.logger.error(f"No best response found for item ID: {item.id}")
            return None

        query = f"""
            UPDATE {catalog}.{schema}.{table_name} 
            SET response = ?, model_id = ?, name = ? 
            WHERE id = ?
        """
        params = (best_response, model_id, name, int(prompt_item.name[:-5]))

        self.logger.info(f"Executing update query for table '{table_name}' with ID '{int(prompt_item.name[:-5])}'.")
        self.logger.debug(f"Update Query Parameters: {params}")

        self.call_databricks_query(server_hostname, databricks_client_id, databricks_client_secret, databricks_http_path, query, params)
        self.logger.info(f"Successfully updated table '{table_name}' for item with ID '{item.id}'.")
        return item
