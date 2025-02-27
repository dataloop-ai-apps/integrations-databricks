# Databricks Integration

This repository provides a service that enables seamless interaction between **Databricks** and **Dataloop** using **OAuth M2M Authentication**. The integration is designed to streamline data processing, table updates, and data uploads between Databricks and Dataloop datasets.

## Features

- **Secure Authentication** with **OAuth M2M Authentication** for Databricks access.
- **SQL Query Execution** on Databricks directly from Dataloop using the integrated service.
- **Dynamic Table Creation and Updates**: Automatically create and update tables based on Dataloop dataset information.
- **Seamless Data Upload**: Upload Databricks query results directly to Dataloop datasets.

## Prerequisites

To set up the integration, you'll need the following information:

- **Server Host Name**
- **Databricks Client ID**
- **Databricks Client Secret**
- **Databricks HTTP Path**
- **Catalog, Schema, and Table Name** in Databricks with at least the following columns:
  - **`id`**: Auto-generated field.
  - **`prompt`**: The prompt to create in Dataloop.
  - **`response`**: Field to store model responses (auto-populated from the RLHF pipeline).
  - **`model_id`**: ID of the model (auto-populated from the RLHF pipeline).
  - **`name`**: Name of the model (auto-populated from the RLHF pipeline).

For detailed instructions on obtaining these prerequisites, refer to the [Databricks OAuth Documentation](https://docs.databricks.com/en/dev-tools/authentication-oauth.html).

## Attributions

This project makes use of the **Databricks SDK for Python**, which is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

## Pipeline Nodes

- **Import Table**
  - This node retrieves prompts from a selected Databricks table and adds them to a specified dataset in Dataloop, creating prompt items accordingly.
- **Export to Table**
  - This node takes the response marked as the best and updates the corresponding Databricks table row with the response, model name and ID from Dataloop.
- **Import Volume**
  - This node transfers all files from a specified Databricks volume into a designated Dataloop dataset, preserving the structure for seamless integration and accessibility.
- **Export item to Volume**
  - This node transfers item from Dataloop into a designated Databricks volume.
