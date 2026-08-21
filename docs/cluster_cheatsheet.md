## Access Credentials

*   **Username:** `[REDACTED - See .env]`
*   **Password:** `[REDACTED - See .env]`
*   **Server Access:** `[REDACTED - See .env]`

## GPU Usage Guidelines

Please be mindful of shared resources to ensure fair access for everyone on the server.

*   **Check Availability:** Before starting any program, run `nvidia-smi` to see which graphics cards are currently available.
*   **Limit Your Computations:** Restrict your jobs to specific GPUs so others can use the remaining ones. You can do this by setting the `CUDA_VISIBLE_DEVICES` environmental variable.
    *   *Example:* To use only the second GPU (index 1), run:
        ```bash
        CUDA_VISIBLE_DEVICES=1
        ```

## Storage & Memory Management

*   **Personal Directory (`/home/[username]`):** Keep your storage footprint as small as possible. Promptly delete trained models or temporary data when they are no longer needed.
*   **Public Datasets:** If you are using public datasets that might benefit other users, store them in the shared `/home/data` directory instead of your personal folder.

> **Note on Fair Use:** Server resources (memory, CPU, GPU) are shared. If we observe that a user is blocking an excessive amount of resources, we may, if necessary, terminate your program to maintain availability for others.