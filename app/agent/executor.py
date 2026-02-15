class PlanExecutor:

    def __init__(self, orchestrator):
        self.orchestrator = orchestrator

    # --------------------------------------------------
    # Execute Multi-Step Plan
    # --------------------------------------------------

    def execute(self, plan_steps):
        """
        Executes each tool step sequentially.
        Returns list of results.
        """

        results = []

        for step in plan_steps:

            tool = step.get("tool")
            parameters = step.get("parameters", {})

            try:
                result = self.orchestrator.handle(tool, parameters)

                results.append({
                    "tool": tool,
                    "status": "success",
                    "result": result
                })

            except Exception as e:
                results.append({
                    "tool": tool,
                    "status": "error",
                    "error": str(e)
                })

                # Stop execution if a step fails
                break

        return results
