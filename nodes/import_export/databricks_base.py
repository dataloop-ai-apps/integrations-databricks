import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import dtlpy as dl
from databricks import sql
from databricks.sdk.core import Config, oauth_service_principal

logger = logging.getLogger(name="databricks-connect")


class DatabricksBase(dl.BaseServiceRunner):
    """
    A class for running a service that interacts with Databricks.
    """

    def __init__(self):
        """
        Initializes the ServiceRunner with Databricks credentials.
        """
        self.logger = logger
        self.max_workers = 5

    def call_databricks_query(
        self,
        server_hostname: str,
        databricks_client_id: str,
        databricks_http_path: str,
        query: str,
        params: tuple = None,
        tempdir: str = '/tmp/',
    ):
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
        self.logger.info(
            "Executing query on server '%s' with HTTP path '%s'.",
            server_hostname,
            databricks_http_path,
        )
        self.logger.debug("Query: %s", query)
        if params:
            self.logger.debug("Query Parameters: %s", params)

        def credential_provider():
            config = Config(
                host=f"https://{server_hostname}",
                client_id=databricks_client_id,
                client_secret=os.environ.get("DATABRICKS_CLIENT_SECRET"),
                product="DatabricksBaseService",
                product_version="1.0",
            )

            config.with_user_agent_extra("integration", "Dataloop")

            return oauth_service_principal(config)

        try:
            with sql.connect(
                server_hostname=server_hostname,
                http_path=databricks_http_path,
                credentials_provider=credential_provider,
                connection_timeout=10,
                staging_allowed_local_path=tempdir,
            ) as connection:
                cursor = connection.cursor()
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)

                # Fetch results for SELECT queries, otherwise commit changes
                if query.strip().upper().startswith(
                    "SELECT"
                ) or query.strip().upper().startswith("LIST"):
                    result = cursor.fetchall()
                else:
                    connection.commit()
                    result = cursor.rowcount

                cursor.close()
                return result

        except Exception as e:
            self.logger.error("Failed to execute query: %s", e)
            raise RuntimeError(f"Failed to execute query: {e}") from e

    def create_table(
        self,
        server_hostname: str,
        databricks_client_id: str,
        databricks_http_path: str,
        catalog: str,
        schema: str,
        table_name: str,
        dataset_id: str,
    ):
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
        self.logger.info(
            "Creating table for dataset '%s' and table '%s'.", dataset_id, table_name
        )

        try:
            dataset = dl.datasets.get(dataset_id=dataset_id)
            self.logger.info("Successfully retrieved dataset with ID '%s'.", dataset_id)
        except Exception as e:
            self.logger.error("Failed to get dataset with ID '%s': %s", dataset_id, e)
            return None

        # Execute query to fetch data
        query = f"SELECT * FROM {catalog}.{schema}.{table_name}"
        result = self.call_databricks_query(
            server_hostname, databricks_client_id, databricks_http_path, query
        )

        prompt_items = []
        for res in result:
            prompt_item = dl.PromptItem(name=str(res.id))
            prompt_item.add(
                message={
                    "role": "user",
                    "content": [{"mimetype": dl.PromptType.TEXT, "value": res.prompt}],
                }
            )
            prompt_items.append(prompt_item)

        # Upload PromptItems to Dataloop
        items = list(dataset.items.upload(local_path=prompt_items, overwrite=True))
        self.logger.info(
            "Successfully uploaded %d items to dataset '%s'.", len(items), dataset_id
        )
        return items

    def update_table(
        self,
        item: dl.Item,
        server_hostname: str,
        databricks_client_id: str,
        databricks_http_path: str,
        catalog: str,
        schema: str,
        table_name: str,
    ):
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
        self.logger.info(
            "Updating table '%s' for item with ID '%s'.", table_name, item.id
        )

        prompt_item = dl.PromptItem.from_item(item)
        first_prompt_key = prompt_item.prompts[0].key

        # Find the best response based on annotation attributes
        best_response = None
        model_id, name = None, "human"  # Default value for 'name' if not found

        for resp in item.annotations.list():
            try:
                is_best = resp.attributes.get("isBest", False)
            except AttributeError:
                is_best = False
            if is_best and resp.metadata["system"].get("promptId") == first_prompt_key:
                best_response = resp.coordinates
                model_info = resp.metadata.get("user", {}).get("model", {})
                model_id = model_info.get("model_id", "")
                name = model_info.get("name", "human")
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

        self.logger.info(
            "Executing update query for table '%s' with ID '%s'.", table_name, item.id
        )
        self.logger.debug("Update Query Parameters: '%s'", params)

        self.call_databricks_query(
            server_hostname,
            databricks_client_id,
            databricks_http_path,
            query,
            params,
        )
        self.logger.info(
            "Successfully updated table '%s' for item with ID '%s'.",
            table_name,
            item.id,
        )
        return item

    def download_file_from_volume(
        self,
        server_hostname: str,
        databricks_client_id: str,
        databricks_http_path: str,
        volume_path: str,
        local_file_path: str,
        tempdir: str,
    ):
        """
        Downloads a file from a Databricks Volume.

        :param server_hostname: Databricks server hostname.
        :param databricks_client_id: Databricks client ID.
        :param databricks_http_path: Databricks HTTP path.
        :param volume_path: Path in the Databricks Volume.
        :param local_file_path: Path to save the downloaded file.
        :param dataset_id: The ID of the dataset to upload the downloaded file.
        """

        local_file_path_temp = os.path.join(tempdir, os.path.basename(local_file_path))
        query = f"GET '{volume_path}' TO '{local_file_path_temp}'"

        self.logger.info(
            "Downloading file from volume '%s' to '%s'.", volume_path, local_file_path
        )
        self.call_databricks_query(
            server_hostname,
            databricks_client_id,
            databricks_http_path,
            query,
            tempdir=tempdir,
        )

        self.logger.info("File downloaded successfully.")

    def upload_dbrx_folder_to_dtlp(
        self,
        server_hostname: str,
        databricks_client_id: str,
        databricks_http_path: str,
        catalog: str,
        schema: str,
        volume_name: str,
        dataset_id: str,
    ):
        """
        Uploads a folder to a Databricks Volume.

        :param server_hostname: Databricks server hostname.
        :param databricks_client_id: Databricks client ID.
        :param databricks_http_path: Databricks HTTP path.
        :param catalog: The Databricks catalog.
        :param schema: The Databricks schema.
        :param volume_name: Name of the Databricks Volume.
        :param dataset_id: The ID of the dataset to upload the downloaded file.

        :return: The uploaded items or None if an error occurs.
        """

        local_folder_path = tempfile.mkdtemp() + "/"
        volume_path = f"/Volumes/{catalog}/{schema}/{volume_name}"

        self.logger.info(
            "Uploading folder '%s' to volume '%s'.", local_folder_path, volume_path
        )

        query = f"LIST '{volume_path}'"
        files = self.call_databricks_query(
            server_hostname,
            databricks_client_id,
            databricks_http_path,
            query,
        )

        self.logger.info("Files in volume: %s", files)

        # Submit tasks and handle exceptions
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for file in files:
                future = executor.submit(
                    self.download_file_from_volume,
                    server_hostname,
                    databricks_client_id,
                    databricks_http_path,
                    file.path,
                    os.path.join(local_folder_path, file.path),
                    local_folder_path,
                )
                futures.append(future)

            # Wait for all tasks and handle exceptions
            for future in as_completed(futures):
                try:
                    future.result()  # Retrieve the result to check for exceptions
                except Exception as e:
                    self.logger.error(f"Task failed with exception: {e}")

        dataset = dl.datasets.get(dataset_id=dataset_id)

        result = dataset.items.upload(local_path=local_folder_path, overwrite=True)

        # Ensure result is iterable, then convert to a list
        items = list(
            result
            if isinstance(result, (list, tuple, set)) or hasattr(result, '__iter__')
            else [result]
        )

        return items

    def upload_item_to_volume(
        self,
        item: dl.Item,
        server_hostname: str,
        databricks_client_id: str,
        databricks_http_path: str,
        catalog: str,
        schema: str,
        volume_name: str,
    ):
        """

        Uploads an item to a Databricks Volume.

        :param item: The Dataloop item to upload.
        :param server_hostname: Databricks server hostname.
        :param databricks_client_id: Databricks client ID.
        :param databricks_http_path: Databricks HTTP path.
        :param catalog: The Databricks catalog.
        :param schema: The Databricks schema.
        :param volume_name: Name of the Databricks Volume.
        """

        file_path = item.download(save_locally=True)
        volume_path = f"/Volumes/{catalog}/{schema}/{volume_name}/{item.name}"
        query = f"PUT '{file_path}' INTO '{volume_path}' OVERWRITE"

        self.logger.info("Uploading item '%s' to volume '%s'.", item.name, volume_path)

        self.call_databricks_query(
            server_hostname,
            databricks_client_id,
            databricks_http_path,
            query,
            tempdir=os.path.dirname(file_path),
        )
        self.logger.info("File uploaded successfully.")

        # Clean up the downloaded file
        os.remove(file_path)

        return item
