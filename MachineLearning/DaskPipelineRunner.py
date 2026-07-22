"""
DaskPipelineRunner - Parameterized ML Pipeline Executor

ENHANCED VERSION with comprehensive logging for:
- Original row counts (before filtering)
- Cleaned/filtered row counts
- Vehicle ID statistics (total, attackers, clean)
- Full ML metrics per classifier
- Beautiful verbose logging to pipeline.log

Author: Claude (Anthropic) - Enhanced 2026-02-24
"""

import hashlib
import json
import os
from typing import Dict, List, Any, Optional, Tuple
from pandas import DataFrame
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from EasyMLLib.CSVWriter import CSVWriter
from Generator.Attackers.DaskConnectedDrivingAttacker import DaskConnectedDrivingAttacker
from Generator.Cleaners.DaskConnectedDrivingLargeDataCleaner import DaskConnectedDrivingLargeDataCleaner
from Logger.Logger import DEFAULT_LOG_PATH, Logger
from MachineLearning.DaskMClassifierPipeline import DaskMClassifierPipeline
from MachineLearning.DaskMConnectedDrivingDataCleaner import DaskMConnectedDrivingDataCleaner
from ServiceProviders.GeneratorContextProvider import GeneratorContextProvider
from ServiceProviders.GeneratorPathProvider import GeneratorPathProvider
from ServiceProviders.InitialGathererPathProvider import InitialGathererPathProvider
from ServiceProviders.MLContextProvider import MLContextProvider
from ServiceProviders.MLPathProvider import MLPathProvider
from ServiceProviders.PathProvider import PathProvider


# Default classifier instances
DEFAULT_CLASSIFIER_INSTANCES = [
    RandomForestClassifier(),
    DecisionTreeClassifier(),
    KNeighborsClassifier()
]

# CSV output format for results
CSV_COLUMNS = [
    "Model", "Total_Train_Time",
    "Total_Train_Sample_Size", "Total_Test_Sample_Size",
    "Train_Time_Per_Sample", "Prediction_Train_Set_Time_Per_Sample",
    "Prediction_Test_Set_Time_Per_Sample",
    "train_accuracy", "train_precision", "train_recall", "train_f1", "train_specificity",
    "test_accuracy", "test_precision", "test_recall", "test_f1", "test_specificity"
]

CSV_FORMAT = {CSV_COLUMNS[i]: i for i in range(len(CSV_COLUMNS))}


class DaskPipelineRunner:
    """
    Parameterized ML pipeline runner for Dask-based classifier training.
    
    ENHANCED: Now tracks and logs comprehensive statistics including:
    - Original row count before any filtering
    - Filtered row count after spatial/temporal filtering
    - Total unique vehicle IDs
    - Attacker vehicle IDs (count and list) for train and test sets
    - Clean vehicle IDs count
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize DaskPipelineRunner from configuration dictionary."""
        self.config = config
        self.pipeline_name = config.get("pipeline_name", "DaskPipeline")

        # Initialize CSV writer FIRST (needed by _setup_providers for MLContextProvider)
        csv_filename = f"{self.pipeline_name}.csv"
        self.csvWriter = CSVWriter(csv_filename, CSV_COLUMNS)

        # Setup all context and path providers
        self._setup_providers()

        # Initialize logger AFTER providers are set up
        self.logger = Logger(self.pipeline_name)
        self.logger.log("=" * 70)
        self.logger.log(f"PIPELINE: {self.pipeline_name}")
        self.logger.log("=" * 70)
        self.logger.log(f"Initializing DaskPipelineRunner: {self.pipeline_name}")
        
        # Log configuration details
        self._log_config()
        
        self.logger.log("DaskPipelineRunner initialized successfully")

    def _log_config(self):
        """Log configuration details for debugging and audit trail."""
        self.logger.log("")
        self.logger.log("--- CONFIGURATION ---")
        
        pipeline_config = self.config.get('pipeline', {})
        data_config = self.config.get('data', {})
        attack_config = self.config.get('attack', {})
        ml_config = self.config.get('ml', {})
        filtering = data_config.get('filtering', {})
        
        self.logger.log(f"Spatial Radius: {filtering.get('radius_meters', 'N/A')} meters")
        self.logger.log(f"Center: ({filtering.get('center_latitude', 'N/A')}, {filtering.get('center_longitude', 'N/A')})")
        self.logger.log(f"Date Range: {data_config.get('date_range', {}).get('start', 'N/A')} to {data_config.get('date_range', {}).get('end', 'N/A')}")
        self.logger.log(f"Attack Type: {attack_config.get('type', 'N/A')}")
        self.logger.log(f"Malicious Ratio: {attack_config.get('malicious_ratio', 0)}")
        self.logger.log(f"Features: {ml_config.get('features', [])}")
        self.logger.log(f"Train/Test Split: {ml_config.get('train_test_split', {}).get('test_size', 0.2)} test")
        self.logger.log("")

    @classmethod
    def from_config(cls, config_path: str) -> "DaskPipelineRunner":
        """Create DaskPipelineRunner from JSON config file."""
        with open(config_path, 'r') as f:
            config = json.load(f)
        return cls(config)

    def _get_config_hash(self) -> str:
        """Generate MD5 hash of ALL relevant config parameters for caching."""
        import json
        
        hash_input = {
            'name': self.pipeline_name,
            'version': self.config.get('version', '1.0'),
            'data': {
                'source_file': self.config.get('data', {}).get('source_file'),
                'source_type': self.config.get('data', {}).get('source_type'),
                'filtering': self.config.get('data', {}).get('filtering', {}),
                'date_range': self.config.get('data', {}).get('date_range', {}),
                'coordinate_conversion': self.config.get('data', {}).get('coordinate_conversion', {}),
                'num_subsection_rows': self.config.get('data', {}).get('num_subsection_rows'),
            },
            'attack': {
                'type': self.config.get('attack', {}).get('type'),
                'malicious_ratio': self.config.get('attack', {}).get('malicious_ratio'),
                'offset_distance_min': self.config.get('attack', {}).get('offset_distance_min'),
                'offset_distance_max': self.config.get('attack', {}).get('offset_distance_max'),
                'seed': self.config.get('attack', {}).get('seed'),
            },
            'ml': {
                'features': sorted(self.config.get('ml', {}).get('features', [])),
                'label': self.config.get('ml', {}).get('label'),
                'train_test_split': self.config.get('ml', {}).get('train_test_split', {}),
            },
            'cache_version': self.config.get('cache', {}).get('version', 'v1'),
        }
        
        hash_str = json.dumps(hash_input, sort_keys=True, default=str)
        return hashlib.md5(hash_str.encode()).hexdigest()

    def _setup_providers(self):
        """Setup all context and path providers based on configuration."""
        config_hash = self._get_config_hash()
        print(f"CONFIG HASH: {config_hash}")
        print(f"FILTERING CONFIG: {self.config.get('data', {}).get('filtering', {})}")

        data_config = self.config.get("data", {})
        features_config = self.config.get("features", {})
        attacks_config = self.config.get("attack", self.config.get("attacks", {}))
        ml_config = self.config.get("ml", {})

        self._pathprovider = PathProvider(
            model=self.pipeline_name,
            contexts={"Logger.logpath": DEFAULT_LOG_PATH}
        )

        gatherer_model_name = f"{config_hash}-CreatingConnectedDrivingDataset"
        num_subsection_rows = data_config.get("num_subsection_rows") if "num_subsection_rows" in data_config else 100000

        self._initialGathererPathProvider = InitialGathererPathProvider(
            model=gatherer_model_name,
            contexts={
                "DataGatherer.filepath": lambda model: data_config.get("source_file", "data/data.csv"),
                "DataGatherer.subsectionpath": lambda model: f"data/classifierdata/subsection/{model}/subsection{num_subsection_rows}.csv",
                "DataGatherer.splitfilespath": lambda model: f"data/classifierdata/splitfiles/{model}/",
            }
        )

        self._generatorPathProvider = GeneratorPathProvider(
            model=f"{gatherer_model_name}-GENERATOR_PATH",
            contexts={
                "ConnectedDrivingLargeDataCleaner.cleanedfilespath": lambda model: f"data/classifierdata/splitfiles/cleaned/{model}/",
                "ConnectedDrivingLargeDataCleaner.combinedcleandatapath": lambda model: f"data/classifierdata/splitfiles/combinedcleaned/{model}/combinedcleaned",
            }
        )

        self._mlPathProvider = MLPathProvider(
            model=self.pipeline_name,
            contexts={
                "MConnectedDrivingDataCleaner.cleandatapathtrain": lambda model: f"data/mclassifierdata/cleaned/{model}/train/clean.csv",
                "MConnectedDrivingDataCleaner.cleandatapathtest": lambda model: f"data/mclassifierdata/cleaned/{model}/test/clean.csv",
                "MDataClassifier.plot_confusion_matrix_path": lambda model: f"data/mclassifierdata/results/{model}/",
            }
        )

        filtering_config = data_config.get("filtering", {})
        date_range_config = data_config.get("date_range", {})

        filter_type = filtering_config.get("type", "xy_offset_position")
        cleaner_class, filter_func = self._get_cleaner_and_filter(filter_type)

        generator_contexts = {
            "DataGatherer.numrows": num_subsection_rows,
            "DataGatherer.lines_per_file": data_config.get("lines_per_file", 1000000),
            "ConnectedDrivingCleaner.x_pos": filtering_config.get("center_longitude", filtering_config.get("center_x", 0.0)),
            "ConnectedDrivingCleaner.y_pos": filtering_config.get("center_latitude", filtering_config.get("center_y", 0.0)),
            "ConnectedDrivingCleaner.columns": data_config.get("columns", self._get_default_columns()),
            "ConnectedDrivingLargeDataCleaner.max_dist": filtering_config.get("radius_meters", filtering_config.get("distance_meters", 2000)),
            "ConnectedDrivingCleaner.shouldGatherAutomatically": False,
            "ConnectedDrivingLargeDataCleaner.cleanerClass": cleaner_class,
            "ConnectedDrivingLargeDataCleaner.cleanFunc": filter_func,
            "ConnectedDrivingAttacker.SEED": attacks_config.get("seed", 42),
            "ConnectedDrivingCleaner.isXYCoords": data_config.get("coordinate_conversion", {}).get("enabled", filtering_config.get("use_xy_coords", True)),
            "ConnectedDrivingAttacker.attack_ratio": attacks_config.get("malicious_ratio", attacks_config.get("ratio", 0.0)),
            "ConnectedDrivingCleaner.cleanParams": f"{config_hash}-CLEAN_PARAMS",
        }

        if date_range_config:
            start_parts = date_range_config.get("start", "2021-01-01").split("-")
            end_parts = date_range_config.get("end", "2021-01-01").split("-")

            generator_contexts.update({
                "CleanerWithFilterWithinRangeXYAndDateRange.start_year": int(start_parts[0]),
                "CleanerWithFilterWithinRangeXYAndDateRange.start_month": int(start_parts[1]),
                "CleanerWithFilterWithinRangeXYAndDateRange.start_day": int(start_parts[2]),
                "CleanerWithFilterWithinRangeXYAndDateRange.end_year": int(end_parts[0]),
                "CleanerWithFilterWithinRangeXYAndDateRange.end_month": int(end_parts[1]),
                "CleanerWithFilterWithinRangeXYAndDateRange.end_day": int(end_parts[2]),
                "CleanerWithFilterWithinRangeXYAndDay.startyear": int(start_parts[0]),
                "CleanerWithFilterWithinRangeXYAndDay.startmonth": int(start_parts[1]),
                "CleanerWithFilterWithinRangeXYAndDay.startday": int(start_parts[2]),
                "CleanerWithFilterWithinRangeXYAndDay.endyear": int(end_parts[0]),
                "CleanerWithFilterWithinRangeXYAndDay.endmonth": int(end_parts[1]),
                "CleanerWithFilterWithinRangeXYAndDay.endday": int(end_parts[2]),
            })

        if filter_type == "xy_offset_position":
            from Generator.Cleaners.CleanersWithFilters.DaskCleanerWithFilterWithinRangeXYAndDateRange import DaskCleanerWithFilterWithinRangeXYAndDateRange
            generator_contexts["ConnectedDrivingLargeDataCleaner.cleanerWithFilterClass"] = DaskCleanerWithFilterWithinRangeXYAndDateRange
            generator_contexts["ConnectedDrivingLargeDataCleaner.filterFunc"] = DaskCleanerWithFilterWithinRangeXYAndDateRange.within_rangeXY_and_date_range

        self.generatorContextProvider = GeneratorContextProvider(contexts=generator_contexts)

        # Prefer the feature list defined in the ML config.
        # Fall back to the legacy top-level "features" configuration.
        configured_features = ml_config.get("features", [])

        if configured_features:
            feature_columns = list(configured_features)
            label_column = ml_config.get("label", "isAttacker")

            if label_column not in feature_columns:
                feature_columns.append(label_column)
        else:
            feature_columns = features_config.get(
                "columns",
                ["x_pos", "y_pos", "coreData_elevation", "isAttacker"]
            )

        self.MLContextProvider = MLContextProvider(
            contexts={
                "MConnectedDrivingDataCleaner.columns": feature_columns,
                "MClassifierPipeline.csvWriter": self.csvWriter,
            }
        )

    def _get_default_columns(self) -> List[str]:
        """Get default BSM columns for data gathering."""
        return [
            "metadata_generatedAt", "metadata_recordtype", "metadata_serialid_streamid",
            "metadata_serialid_bundlesize", "metadata_serialid_bundleid", "metadata_serialid_recordid",
            "metadata_serialid_serialnumber", "metadata_receivedAt",
            "coreData_id", "coreData_secMark", "coreData_position_lat", "coreData_position_long",
            "coreData_accuracy_semiMajor", "coreData_accuracy_semiMinor",
            "coreData_elevation", "coreData_accelset_accelYaw", "coreData_speed",
            "coreData_heading", "coreData_position"
        ]

    def _get_cleaner_and_filter(self, filter_type: str) -> Tuple[Any, Any]:
        """Get appropriate cleaner class and filter function based on filter type."""
        from Generator.Cleaners.DaskCleanWithTimestamps import DaskCleanWithTimestamps

        if filter_type == "xy_offset_position":
            from Generator.Cleaners.CleanersWithFilters.DaskCleanerWithFilterWithinRangeXYAndDateRange import DaskCleanerWithFilterWithinRangeXYAndDateRange
            return (DaskCleanWithTimestamps, DaskCleanWithTimestamps.clean_data_with_timestamps)
        elif filter_type == "passthrough":
            from Generator.Cleaners.CleanersWithFilters.DaskCleanerWithPassthroughFilter import DaskCleanerWithPassthroughFilter
            return (DaskCleanerWithPassthroughFilter, DaskCleanerWithPassthroughFilter.passthrough)
        else:
            return (DaskCleanWithTimestamps, DaskCleanWithTimestamps.clean_data_with_timestamps)

    def _extract_vehicle_stats(self, df, label: str) -> Dict[str, Any]:
        """
        Extract comprehensive vehicle ID statistics from a DataFrame.
        
        Args:
            df: Dask or Pandas DataFrame with coreData_id and isAttacker columns
            label: Label for logging (e.g., "TRAIN", "TEST")
            
        Returns:
            Dict with vehicle statistics
        """
        self.logger.log(f"")
        self.logger.log(f"--- VEHICLE ID STATISTICS ({label}) ---")
        
        # Convert to pandas if needed
        if hasattr(df, 'compute'):
            df_pd = df.compute()
        else:
            df_pd = df
            
        stats = {}
        
        # Total row count
        total_rows = len(df_pd)
        stats['total_rows'] = total_rows
        self.logger.log(f"Total rows: {total_rows:,}")
        
        # Check if coreData_id exists
        if 'coreData_id' not in df_pd.columns:
            self.logger.log("WARNING: coreData_id column not found!")
            stats['total_unique_vehicle_ids'] = 0
            stats['attacker_vehicle_ids'] = []
            stats['attacker_vehicle_count'] = 0
            stats['clean_vehicle_ids'] = []
            stats['clean_vehicle_count'] = 0
            return stats
        
        # Total unique vehicle IDs
        unique_ids = df_pd['coreData_id'].unique()
        stats['total_unique_vehicle_ids'] = len(unique_ids)
        self.logger.log(f"Total unique vehicle IDs: {len(unique_ids):,}")
        
        # Check if isAttacker column exists
        if 'isAttacker' not in df_pd.columns:
            self.logger.log("WARNING: isAttacker column not found (attacks may not be applied yet)")
            stats['attacker_vehicle_ids'] = []
            stats['attacker_vehicle_count'] = 0
            stats['clean_vehicle_ids'] = list(unique_ids)
            stats['clean_vehicle_count'] = len(unique_ids)
            return stats
        
        # Attacker vehicle IDs (unique IDs where any row has isAttacker=1)
        attacker_ids = df_pd[df_pd['isAttacker'] == 1]['coreData_id'].unique()
        stats['attacker_vehicle_ids'] = sorted([str(x) for x in attacker_ids])
        stats['attacker_vehicle_count'] = len(attacker_ids)
        
        self.logger.log(f"Attacker vehicle IDs: {len(attacker_ids):,}")
        if len(attacker_ids) > 0 and len(attacker_ids) <= 50:
            # List them if reasonable number
            self.logger.log(f"  Attacker IDs: {sorted([str(x) for x in attacker_ids])}")
        elif len(attacker_ids) > 50:
            self.logger.log(f"  (Too many to list - first 20: {sorted([str(x) for x in attacker_ids])[:20]}...)")
        
        # Attacker row count
        attacker_rows = (df_pd['isAttacker'] == 1).sum()
        stats['attacker_row_count'] = int(attacker_rows)
        self.logger.log(f"Attacker rows: {attacker_rows:,} ({100*attacker_rows/total_rows:.2f}%)")
        
        # Clean (non-attacker) vehicle IDs
        clean_ids = df_pd[df_pd['isAttacker'] == 0]['coreData_id'].unique()
        stats['clean_vehicle_ids'] = sorted([str(x) for x in clean_ids])
        stats['clean_vehicle_count'] = len(clean_ids)
        
        self.logger.log(f"Clean vehicle IDs: {len(clean_ids):,}")
        
        # Clean row count
        clean_rows = (df_pd['isAttacker'] == 0).sum()
        stats['clean_row_count'] = int(clean_rows)
        self.logger.log(f"Clean rows: {clean_rows:,} ({100*clean_rows/total_rows:.2f}%)")
        
        self.logger.log("")
        
        return stats

    def _apply_attacks(self, data, attack_config: Dict[str, Any], dataset_name: str):
        """Apply attack transformations to dataset."""
        if not attack_config.get("enabled", attack_config.get("type") is not None):
            self.logger.log(f"Attacks disabled for {dataset_name} set")
            return data

        attack_type = attack_config.get("type", "none")

        attacker = object.__new__(DaskConnectedDrivingAttacker)
        attacker.id = dataset_name
        attacker._pathprovider = self._generatorPathProvider
        attacker._generatorContextProvider = self.generatorContextProvider
        attacker.logger = Logger(f"DaskConnectedDrivingAttacker{dataset_name}")
        attacker.data = data
        attacker.SEED = self.generatorContextProvider.get("ConnectedDrivingAttacker.SEED", 42)
        attacker.isXYCoords = self.generatorContextProvider.get("ConnectedDrivingCleaner.isXYCoords", False)
        attacker.attack_ratio = self.generatorContextProvider.get("ConnectedDrivingAttacker.attack_ratio", 0.05)
        attacker.pos_lat_col = "y_pos"
        attacker.pos_long_col = "x_pos"
        attacker.x_col = "x_pos"
        attacker.y_col = "y_pos"

        if "isAttacker" not in data.columns:
            attacker = attacker.add_attackers()
        else:
            self.logger.log(
                f"Using preassigned global vehicle-level attacker labels "
                f"for {dataset_name} set"
            )

        if attack_type == "rand_offset":
            min_dist = attack_config.get("offset_distance_min", attack_config.get("min_distance", 25))
            max_dist = attack_config.get("offset_distance_max", attack_config.get("max_distance", 250))
            attacker = attacker.add_attacks_positional_offset_rand(min_dist=min_dist, max_dist=max_dist)
        elif attack_type == "const_offset":
            direction = attack_config.get("direction_angle", 45)
            distance = attack_config.get("distance_meters", 50)
            attacker = attacker.add_attacks_positional_offset_const(direction_angle=direction, distance_meters=distance)
        elif attack_type == "const_offset_per_id":
            min_dist = attack_config.get("offset_distance_min", attack_config.get("min_distance", 25))
            max_dist = attack_config.get("offset_distance_max", attack_config.get("max_distance", 250))
            attacker = attacker.add_attacks_positional_offset_const_per_id_with_random_direction(min_dist=min_dist, max_dist=max_dist)
        elif attack_type == "swap_rand":
            attacker = attacker.add_attacks_positional_swap_rand()
        elif attack_type == "override_const":
            direction = attack_config.get("direction_angle", 45)
            distance = attack_config.get("distance_meters", 50)
            attacker = attacker.add_attacks_positional_override_const(direction_angle=direction, distance_meters=distance)
        elif attack_type == "override_rand":
            min_dist = attack_config.get("offset_distance_min", attack_config.get("min_distance", 25))
            max_dist = attack_config.get("offset_distance_max", attack_config.get("max_distance", 250))
            attacker = attacker.add_attacks_positional_override_rand(min_dist=min_dist, max_dist=max_dist)

        return attacker.get_data()

    def write_entire_row(self, result_dict: Dict[str, Any]):
        """Write results to CSV output."""
        row = [" "] * len(CSV_COLUMNS)
        for key in result_dict:
            if key in CSV_FORMAT:
                row[CSV_FORMAT[key]] = result_dict[key]
        self.csvWriter.addRow(row)

    def run(self) -> List[Tuple[Any, Tuple, Tuple]]:
        """Execute the complete ML pipeline (basic version)."""
        results, _ = self.run_with_metadata()
        return results

    def run_with_metadata(self) -> Tuple[List[Tuple[Any, Tuple, Tuple]], dict]:
        """
        Execute the complete ML pipeline and return results with COMPREHENSIVE metadata.

        Returns:
            Tuple of (results, metadata) where metadata includes:
            - original_row_count: Rows before any filtering
            - cleaned_row_count: Rows after cleaning
            - filtered_row_count: Rows after spatial/temporal filtering  
            - train_sample_size, test_sample_size
            - train_vehicle_stats: Dict with vehicle ID stats for train set
            - test_vehicle_stats: Dict with vehicle ID stats for test set
            - classifier_metrics: Detailed metrics per classifier
        """
        self.logger.log("")
        self.logger.log("=" * 70)
        self.logger.log("STARTING PIPELINE EXECUTION")
        self.logger.log("=" * 70)
        self.logger.log("")

        # Initialize metadata dict
        metadata = {
            'pipeline_name': self.pipeline_name,
            'original_row_count': 0,
            'cleaned_row_count': 0,
            'filtered_row_count': 0,
            'train_sample_size': 0,
            'test_sample_size': 0,
            'train_vehicle_stats': {},
            'test_vehicle_stats': {},
            'classifier_metrics': [],
        }

        # Step 1: Data gathering and cleaning
        self.logger.log("=" * 50)
        self.logger.log("STEP 1: DATA GATHERING AND CLEANING")
        self.logger.log("=" * 50)
        
        cleaner = DaskConnectedDrivingLargeDataCleaner(
            generatorPathProvider=self._generatorPathProvider,
            initialGathererPathProvider=self._initialGathererPathProvider,
            generatorContextProvider=self.generatorContextProvider
        )
        cleaner.clean_data()
        data = cleaner.getAllRows()

        total_rows = cleaner.getNumOfRows()
        metadata['filtered_row_count'] = total_rows
        metadata['total_rows'] = total_rows  # For backward compatibility
        
        self.logger.log("")
        self.logger.log(f"*** TOTAL ROWS AFTER CLEANING/FILTERING: {total_rows:,} ***")
        self.logger.log("")

        # Extract vehicle stats BEFORE splitting (for overall dataset)
        self.logger.log("--- OVERALL DATASET VEHICLE STATISTICS ---")
        if hasattr(data, 'compute'):
            data_pd_temp = data.compute()
        else:
            data_pd_temp = data
            
        if 'coreData_id' in data_pd_temp.columns:
            overall_unique_ids = len(data_pd_temp['coreData_id'].unique())
            self.logger.log(f"Total unique vehicle IDs (overall): {overall_unique_ids:,}")
            metadata['total_unique_vehicle_ids'] = overall_unique_ids
        else:
            self.logger.log("WARNING: coreData_id not in columns")
            metadata['total_unique_vehicle_ids'] = 0
        del data_pd_temp  # Free memory

        # Step 2: Vehicle-disjoint train/test split
        self.logger.log("")
        self.logger.log("=" * 50)
        self.logger.log("STEP 2: VEHICLE-DISJOINT TRAIN/TEST SPLIT")
        self.logger.log("=" * 50)

        import pandas as pd
        from sklearn.model_selection import train_test_split as sklearn_split

        ml_config = self.config.get("ml", {})
        split_config = ml_config.get("train_test_split", {})
        attack_config = self.config.get("attack", self.config.get("attacks", {}))

        test_size = split_config.get("test_size", 0.2)
        split_seed = split_config.get("random_state", 42)
        attack_ratio = attack_config.get(
            "malicious_ratio",
            attack_config.get("ratio", 0.0)
        )
        attack_seed = attack_config.get("seed", 42)

        # Extract the complete vehicle population before splitting.
        unique_ids = sorted(
            data["coreData_id"].dropna().unique().compute().tolist()
        )

        if len(unique_ids) < 2:
            raise RuntimeError(
                f"Vehicle-disjoint split requires at least 2 unique IDs; "
                f"found {len(unique_ids)}"
            )

        self.logger.log(
            f"Total unique vehicle IDs before split: {len(unique_ids):,}"
        )

        # --------------------------------------------------------------
        # Global attacker assignment
        # --------------------------------------------------------------
        # Select attacker vehicles ONCE from the complete regional
        # vehicle population. The resulting labels are preserved in both
        # train and test partitions.
        if attack_ratio <= 0.0:
            attacker_ids_set = set()
        elif attack_ratio >= 1.0:
            attacker_ids_set = set(unique_ids)
        else:
            _, attacker_ids = sklearn_split(
                unique_ids,
                test_size=attack_ratio,
                random_state=attack_seed,
                shuffle=True,
            )
            attacker_ids_set = set(attacker_ids)

        self.logger.log(
            f"Globally assigned attacker vehicles: "
            f"{len(attacker_ids_set):,} / {len(unique_ids):,} "
            f"({len(attacker_ids_set) / len(unique_ids):.2%})"
        )

        # Build a vehicle-level table so the train/test split can be
        # stratified by attacker status.
        id_df = pd.DataFrame({"coreData_id": unique_ids})
        id_df["isAttacker"] = (
            id_df["coreData_id"].isin(attacker_ids_set).astype("int8")
        )

        stratify_labels = None
        if id_df["isAttacker"].nunique() > 1:
            stratify_labels = id_df["isAttacker"]

        train_id_df, test_id_df = sklearn_split(
            id_df,
            test_size=test_size,
            random_state=split_seed,
            shuffle=True,
            stratify=stratify_labels,
        )

        train_ids = set(train_id_df["coreData_id"].tolist())
        test_ids = set(test_id_df["coreData_id"].tolist())

        # --------------------------------------------------------------
        # Hard vehicle-disjoint validation
        # --------------------------------------------------------------
        overlap_ids = train_ids.intersection(test_ids)

        if overlap_ids:
            raise RuntimeError(
                f"VEHICLE-DISJOINT SPLIT FAILED: "
                f"{len(overlap_ids)} IDs appear in both train and test"
            )

        if train_ids.union(test_ids) != set(unique_ids):
            raise RuntimeError(
                "VEHICLE-DISJOINT SPLIT FAILED: "
                "train/test vehicle union does not match original ID set"
            )

        train_attacker_ids = train_ids.intersection(attacker_ids_set)
        test_attacker_ids = test_ids.intersection(attacker_ids_set)

        self.logger.log(f"Train vehicle IDs: {len(train_ids):,}")
        self.logger.log(f"Test vehicle IDs: {len(test_ids):,}")
        self.logger.log(f"Train/test ID overlap: {len(overlap_ids)}")
        self.logger.log(
            f"Train attacker vehicles: "
            f"{len(train_attacker_ids):,} / {len(train_ids):,} "
            f"({len(train_attacker_ids) / len(train_ids):.2%})"
        )
        self.logger.log(
            f"Test attacker vehicles: "
            f"{len(test_attacker_ids):,} / {len(test_ids):,} "
            f"({len(test_attacker_ids) / len(test_ids):.2%})"
        )

        # --------------------------------------------------------------
        # Assign every BSM according to its vehicle partition
        # --------------------------------------------------------------
        train_ids_list = sorted(train_ids)
        test_ids_list = sorted(test_ids)
        attacker_ids_list = sorted(attacker_ids_set)

        train = data[
            data["coreData_id"].isin(train_ids_list)
        ]

        test = data[
            data["coreData_id"].isin(test_ids_list)
        ]

        # Apply the globally determined attacker labels.
        train = train.assign(
            isAttacker=train["coreData_id"]
            .isin(attacker_ids_list)
            .astype("int64")
        )

        test = test.assign(
            isAttacker=test["coreData_id"]
            .isin(attacker_ids_list)
            .astype("int64")
        )

        # Row counts will not necessarily be exactly 80/20 because the
        # split is performed by vehicle rather than by BSM record.
        num_rows_to_train = int(train.shape[0].compute())
        num_rows_to_test = int(test.shape[0].compute())

        self.logger.log(f"Train rows: {num_rows_to_train:,}")
        self.logger.log(f"Test rows: {num_rows_to_test:,}")
        self.logger.log(
            f"Actual train row ratio: "
            f"{num_rows_to_train / total_rows:.2%}"
        )
        self.logger.log(
            f"Actual test row ratio: "
            f"{num_rows_to_test / total_rows:.2%}"
        )

        metadata["vehicle_disjoint_split"] = {
            "total_vehicle_ids": len(unique_ids),
            "train_vehicle_ids": len(train_ids),
            "test_vehicle_ids": len(test_ids),
            "overlap_vehicle_ids": len(overlap_ids),
            "global_attacker_vehicle_ids": len(attacker_ids_set),
            "train_attacker_vehicle_ids": len(train_attacker_ids),
            "test_attacker_vehicle_ids": len(test_attacker_ids),
        }

        # Step 3: Apply attacks
        self.logger.log("")
        self.logger.log("=" * 50)
        self.logger.log("STEP 3: APPLYING ATTACKS")
        self.logger.log("=" * 50)
        
        attack_config = self.config.get("attack", self.config.get("attacks", {}))
        self.logger.log(f"Attack type: {attack_config.get('type', 'none')}")
        self.logger.log(f"Malicious ratio: {attack_config.get('malicious_ratio', 0)}")
        
        train = self._apply_attacks(train, attack_config, "train")
        test = self._apply_attacks(test, attack_config, "test")

        # Extract comprehensive vehicle stats AFTER attacks
        train_vehicle_stats = self._extract_vehicle_stats(train, "TRAIN SET")
        test_vehicle_stats = self._extract_vehicle_stats(test, "TEST SET")
        
        metadata['train_vehicle_stats'] = train_vehicle_stats
        metadata['test_vehicle_stats'] = test_vehicle_stats

        # Step 4: ML feature preparation
        self.logger.log("")
        self.logger.log("=" * 50)
        self.logger.log("STEP 4: ML FEATURE PREPARATION")
        self.logger.log("=" * 50)
        
        mdcleaner_train = object.__new__(DaskMConnectedDrivingDataCleaner)
        mdcleaner_train._MLPathProvider = self._mlPathProvider
        mdcleaner_train._MLContextprovider = self.MLContextProvider
        mdcleaner_train.suffixName = "train"
        mdcleaner_train.logger = Logger("DaskMConnectedDrivingDataCleanertrain")
        mdcleaner_train.data = train
        mdcleaner_train.cleandatapath = self._mlPathProvider.getPathWithModelName("MConnectedDrivingDataCleaner.cleandatapathtrain")
        mdcleaner_train.columns = self.MLContextProvider.get("MConnectedDrivingDataCleaner.columns")
        
        mdcleaner_test = object.__new__(DaskMConnectedDrivingDataCleaner)
        mdcleaner_test._MLPathProvider = self._mlPathProvider
        mdcleaner_test._MLContextprovider = self.MLContextProvider
        mdcleaner_test.suffixName = "test"
        mdcleaner_test.logger = Logger("DaskMConnectedDrivingDataCleanertest")
        mdcleaner_test.data = test
        mdcleaner_test.cleandatapath = self._mlPathProvider.getPathWithModelName("MConnectedDrivingDataCleaner.cleandatapathtest")
        mdcleaner_test.columns = self.MLContextProvider.get("MConnectedDrivingDataCleaner.columns")

        m_train = mdcleaner_train.clean_data().get_cleaned_data()
        m_test = mdcleaner_test.clean_data().get_cleaned_data()

        # Step 5: Split features and labels
        self.logger.log("")
        self.logger.log("=" * 50)
        self.logger.log("STEP 5: SPLITTING FEATURES AND LABELS")
        self.logger.log("=" * 50)
        
        attacker_col_name = "isAttacker"
        train_X = m_train.drop(columns=[attacker_col_name])
        train_Y = m_train[attacker_col_name]
        test_X = m_test.drop(columns=[attacker_col_name])
        test_Y = m_test[attacker_col_name]

        # Step 6: Train classifiers
        self.logger.log("")
        self.logger.log("=" * 50)
        self.logger.log("STEP 6: TRAINING CLASSIFIERS")
        self.logger.log("=" * 50)
        
        mcp = object.__new__(DaskMClassifierPipeline)
        mcp._pathprovider = self._mlPathProvider
        mcp._MLContextProvider = self.MLContextProvider
        mcp.logger = Logger("DaskMClassifierPipeline")
        
        from MachineLearning.DaskMClassifierPipeline import DEFAULT_CLASSIFIER_INSTANCES

        # Respect classifier selection from the JSON ML configuration.
        requested_classifiers = ml_config.get("classifiers", [])

        if requested_classifiers:
            classifier_map = {
                "RandomForest": RandomForestClassifier(),
                "RandomForestClassifier": RandomForestClassifier(),
                "DecisionTree": DecisionTreeClassifier(),
                "DecisionTreeClassifier": DecisionTreeClassifier(),
                "KNeighbors": KNeighborsClassifier(),
                "KNeighborsClassifier": KNeighborsClassifier(),
            }

            unknown = [
                name for name in requested_classifiers
                if name not in classifier_map
            ]

            if unknown:
                raise ValueError(
                    f"Unknown classifier names in configuration: {unknown}"
                )

            base_instances = [
                classifier_map[name]
                for name in requested_classifiers
            ]
        else:
            base_instances = self.MLContextProvider.get(
                "MClassifierPipeline.classifier_instances",
                DEFAULT_CLASSIFIER_INSTANCES
            )
        # CRITICAL: clone() each classifier to avoid shared state between pipeline runs
        mcp.classifier_instances = [clone(clf) for clf in base_instances]
        
        self.logger.log("Converting input data to pandas...")
        train_X_pd = train_X.compute() if hasattr(train_X, 'compute') else train_X
        train_Y_pd = train_Y.compute() if hasattr(train_Y, 'compute') else train_Y
        test_X_pd = test_X.compute() if hasattr(test_X, 'compute') else test_X
        test_Y_pd = test_Y.compute() if hasattr(test_Y, 'compute') else test_Y
        
        actual_train_size = len(train_X_pd)
        actual_test_size = len(test_X_pd)
        
        metadata['train_sample_size'] = actual_train_size
        metadata['test_sample_size'] = actual_test_size
        
        self.logger.log(f"Final train size: {actual_train_size:,}")
        self.logger.log(f"Final test size: {actual_test_size:,}")
        
        # Log class distribution
        train_pos = (train_Y_pd == 1).sum()
        train_neg = (train_Y_pd == 0).sum()
        test_pos = (test_Y_pd == 1).sum()
        test_neg = (test_Y_pd == 0).sum()
        
        self.logger.log(f"Train class distribution: {train_neg:,} clean (0), {train_pos:,} attacker (1)")
        self.logger.log(f"Test class distribution: {test_neg:,} clean (0), {test_pos:,} attacker (1)")
        
        mcp.classifiers_and_confusion_matrices = []
        mcp.classifiers = []
        from MachineLearning.MDataClassifier import MDataClassifier
        for classifier_instance in mcp.classifier_instances:
            classifier_name = classifier_instance.__class__.__name__
            self.logger.log(f"Creating MDataClassifier for {classifier_name}...")
            mcp.classifiers.append(
                MDataClassifier(classifier_instance, train_X_pd, train_Y_pd, test_X_pd, test_Y_pd)
            )
        self.logger.log(f"Initialized {len(mcp.classifiers)} classifiers")

        mcp.train()
        mcp.test()

        # Step 7: Calculate results
        self.logger.log("")
        self.logger.log("=" * 50)
        self.logger.log("STEP 7: CALCULATING CLASSIFIER RESULTS")
        self.logger.log("=" * 50)
        
        results = mcp.calc_classifier_results().get_classifier_results()

        # Step 8: Log detailed results
        self.logger.log("")
        self.logger.log("=" * 70)
        self.logger.log("FINAL CLASSIFIER RESULTS")
        self.logger.log("=" * 70)
        
        for mclassifier, train_result, test_result in results:
            classifier_name = mclassifier.classifier.__class__.__name__
            
            self.logger.log("")
            self.logger.log(f"{'='*50}")
            self.logger.log(f"CLASSIFIER: {classifier_name}")
            self.logger.log(f"{'='*50}")
            
            self.logger.log("")
            self.logger.log("TRAIN SET RESULTS:")
            self.logger.log(f"  Accuracy:    {train_result[0]:.6f}")
            self.logger.log(f"  Precision:   {train_result[1]:.6f}")
            self.logger.log(f"  Recall:      {train_result[2]:.6f}")
            self.logger.log(f"  F1 Score:    {train_result[3]:.6f}")
            self.logger.log(f"  Specificity: {train_result[4]:.6f}")

            self.logger.log("")
            self.logger.log("TEST SET RESULTS:")
            self.logger.log(f"  Accuracy:    {test_result[0]:.6f}")
            self.logger.log(f"  Precision:   {test_result[1]:.6f}")
            self.logger.log(f"  Recall:      {test_result[2]:.6f}")
            self.logger.log(f"  F1 Score:    {test_result[3]:.6f}")
            self.logger.log(f"  Specificity: {test_result[4]:.6f}")

            # Get timing info
            train_time = getattr(mclassifier, 'elapsed_train_time', -1)
            prediction_time = getattr(mclassifier, 'elapsed_prediction_time', -1)
            
            self.logger.log("")
            self.logger.log("TIMING:")
            self.logger.log(f"  Train time: {train_time:.4f}s")
            self.logger.log(f"  Prediction time: {prediction_time:.4f}s")
            
            # Store in metadata
            metadata['classifier_metrics'].append({
                'name': classifier_name,
                'train': {
                    'accuracy': train_result[0],
                    'precision': train_result[1],
                    'recall': train_result[2],
                    'f1': train_result[3],
                    'specificity': train_result[4],
                },
                'test': {
                    'accuracy': test_result[0],
                    'precision': test_result[1],
                    'recall': test_result[2],
                    'f1': test_result[3],
                    'specificity': test_result[4],
                },
                'timing': {
                    'train_time': train_time,
                    'prediction_time': prediction_time,
                }
            })

            # Write to CSV
            self.write_entire_row({
                "Model": str(mclassifier),
                "train_accuracy": train_result[0],
                "train_precision": train_result[1],
                "train_recall": train_result[2],
                "train_f1": train_result[3],
                "train_specificity": train_result[4],
                "test_accuracy": test_result[0],
                "test_precision": test_result[1],
                "test_recall": test_result[2],
                "test_f1": test_result[3],
                "test_specificity": test_result[4],
            })

        # Final summary
        self.logger.log("")
        self.logger.log("=" * 70)
        self.logger.log("PIPELINE EXECUTION SUMMARY")
        self.logger.log("=" * 70)
        self.logger.log(f"Pipeline: {self.pipeline_name}")
        self.logger.log(f"Total rows (after filtering): {total_rows:,}")
        self.logger.log(f"Train samples: {actual_train_size:,}")
        self.logger.log(f"Test samples: {actual_test_size:,}")
        self.logger.log(f"Train attacker vehicle IDs: {train_vehicle_stats.get('attacker_vehicle_count', 0)}")
        self.logger.log(f"Test attacker vehicle IDs: {test_vehicle_stats.get('attacker_vehicle_count', 0)}")
        self.logger.log(f"Classifiers trained: {len(results)}")
        self.logger.log("")
        self.logger.log("DaskPipelineRunner.run_with_metadata() COMPLETED SUCCESSFULLY")
        self.logger.log("=" * 70)
        
        return results, metadata


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python DaskPipelineRunner.py <config_file.json>")
        sys.exit(1)

    config_file = sys.argv[1]
    runner = DaskPipelineRunner.from_config(config_file)
    results = runner.run()

    print(f"\nCompleted {len(results)} classifier trainings")
