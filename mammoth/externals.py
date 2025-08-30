def safeexec(code: str, out: str = "commons", whitelist: list[str] = None):
    if whitelist is None:
        whitelist = []

    if code.endswith(".py"):
        with open(code, "r") as file:
            code = file.read()

    # Check for disallowed imports
    for line in code.splitlines():
        line = line.strip()
        if line.startswith("import "):
            imported_modules = line.split("import", 1)[1].strip().split(",")
            for module in imported_modules:
                module_name = module.split()[0].split(".")[0].strip()
                if module_name not in whitelist:
                    raise Exception(
                        f"Disallowed import detected: '{module_name}'. Only these are allowed: {whitelist}"
                    )
        elif line.startswith("from "):
            parts = line.split()
            if len(parts) > 1:
                module_name = parts[1].split(".")[0]
                if module_name not in whitelist:
                    raise Exception(
                        f"Disallowed import detected: '{module_name}'. Only these are allowed: {whitelist}"
                    )

    # Execute the code in a local context
    exec_context = locals().copy()
    exec(code, exec_context)
    return exec_context[out]
