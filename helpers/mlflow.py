import mlflow
from mlflow import MlflowClient
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
from enum import Enum
import pandas as pd

runs_uri = "runs:/{}/model"


class ModelAlias(Enum):
    STAGING = "staging"
    ARCHIVED = "archived"
    PRODUCTION = "production"


class Mlflow:

    MLFLOW_TRACKING_URI = "http://localhost:8080"

    def __init__(
        self,
        model_name,
        model_version=None,
    ):
        print("Setting connection to Mlflow...")
        self.client = MlflowClient(
            tracking_uri=self.MLFLOW_TRACKING_URI,
            registry_uri="/home/manpm/Developers/kaggle",
        )
        mlflow.set_tracking_uri(self.MLFLOW_TRACKING_URI)
        self.model_name = model_name
        self.model_version = model_version
        return None

    def get_model_info(
        self, alias: str = ""
    ) -> mlflow.entities.model_registry.ModelVersion:
        if self.model_name == None or alias == "":
            return None
        return self.client.get_model_version_by_alias(self.model_name, alias)

    def register_model(self) -> bool:
        print(self.model_name)
        try:
            self.client.create_registered_model(self.model_name)
        except Exception as e:
            print(f"{self.model_name} has been existed")

    def version_model(self, run_id, model_desc: str = "") -> dict:
        run_uri = runs_uri.format(run_id)
        model_src = RunsArtifactRepository.get_underlying_uri(run_uri)
        mv = self.client.create_model_version(
            self.model_name, model_src, run_id, description=model_desc
        )
        return mv

    def get_or_create_exp(
        self,
        project_name: str,
        experiment_name: str,
        experiment_description: str,
    ):
        experiment_tags = {
            "project_name": project_name,
            "store_dept": "produce",
            "team": "be-ds",
            "project_quarter": "Q1-2024",
            "mlflow.note.content": experiment_description,
        }
        mlflow_experiment = self.client.get_experiment_by_name(experiment_name)
        print("Checking whether experiment existed")
        # Check if exp existed
        if mlflow_experiment == None:
            print("Created a new exp")
            experiment_id = self.client.create_experiment(
                name=experiment_name, tags=experiment_tags
            )
            experiment = mlflow.get_experiment(experiment_id)
            experiment_name = experiment.name
        print(f"Using existed experiment: {experiment_name}")
        # Set current exp
        mlflow.set_experiment(experiment_name)
        return experiment_name

    def load_model(self, run_uri, version, dst_path=None):
        dst_path = dst_path or f"{self.model_name}_{version}"
        local_path = mlflow.artifacts.download_artifacts(
            artifact_uri=f"runs:/{run_uri}/model/model.lgb",
            dst_path=dst_path,
        )
        return local_path

    def stage_model(self, mode_alias, model_version: str) -> bool:
        try:
            self.client.set_registered_model_alias(
                self.model_name, mode_alias, f"{model_version}"
            )
            return True
        except Exception as e:
            print(
                f"Failed to stage {self.model_name} to alias: {mode_alias}. Detail: {e}"
            )
            return False

    def load_prod_model(self):
        mode_info = self.get_model_info(alias=ModelAlias.PRODUCTION.value)
        production_run_uri = mode_info.run_id
        production_version = mode_info.version
        print(f"Loading model {self.model_name} version {mode_info.version}")
        return self.load_model(run_uri=production_run_uri, version=production_version)

    def to_production(self, version):
        prod_success = False
        # Get production model
        prod_model_info = self.get_model_info(alias=ModelAlias.PRODUCTION.value)
        print(f"Production model version: {prod_model_info.version}")
        if int(version) == int(prod_model_info.version):
            print("Current model has been production version so far.")
            return prod_success
        # Archive it
        archived_success = self.stage_model(
            f"{ModelAlias.ARCHIVED.value}_{prod_model_info.version}",
            prod_model_info.version,
        )
        print("Archived")
        if archived_success:
            # Move current model to prod
            prod_success = self.stage_model(ModelAlias.PRODUCTION.value, f"{version}")
            print(f"Production model: {version}")

        return prod_success


def viz_feature_important(clf):
    feature_important = clf.get_booster().get_score(importance_type="weight")
    keys = list(feature_important.keys())
    values = list(feature_important.values())

    data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(
        by="score", ascending=False
    )
    data.nlargest(40, columns="score").plot(
        kind="barh", figsize=(20, 10)
    )  ## plot top 40 features
