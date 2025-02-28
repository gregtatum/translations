import os
import wandb
import wandb_workspaces.workspaces as ws
import wandb_workspaces.reports.v2 as wr

wandb.login()

project = "moz-translations"


def main() -> None:
    workspace = ws.Workspace(
        project=project,
        entity="en-cs",
        name="spring-2024",
        sections=[
            ws.Section(
                name="Validation Metrics",
                panels=[
                    wr.LinePlot(x="Step", y=["val_loss"]),
                    wr.BarPlot(metrics=["val_accuracy"]),
                    wr.ScalarChart(metric="f1_score", groupby_aggfunc="mean"),
                ],
                is_open=True,
            ),
        ],
    )
    print("!!! workspace", workspace)
    a = [
        "Config",
        "Metric",
        "Ordering",
        "RunSettings",
        "RunsetSettings",
        "Section",
        "SectionLayoutSettings",
        "SectionPanelSettings",
        "Summary",
        "Tags",
        "Workspace",
        "WorkspaceSettings",
    ]


if __name__ == "__main__":
    main()
