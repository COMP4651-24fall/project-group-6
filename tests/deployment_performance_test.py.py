from typing import List, Dict, Optional, Any
import requests
import time
import concurrent.futures
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt
from datetime import datetime
import yaml
from pathlib import Path


class DeploymentType(Enum):
    OPENFAAS = "openfaas"
    TRADITIONAL = "traditional"


@dataclass
class TestResult:
    deployment_type: DeploymentType
    latency: float
    status_code: int
    response: Optional[Dict[str, Any]]
    timestamp: float


@dataclass
class TestConfig:
    name: str
    endpoint_url: str
    deployment_type: DeploymentType
    headers: Optional[Dict[str, str]] = None


class ModelPerformanceTester:
    def __init__(self, configs: List[TestConfig]) -> None:
        """
        Initialize the performance tester with multiple deployment configurations.

        Args:
            configs (List[TestConfig]): List of test configurations for different deployments
        """
        self.configs: List[TestConfig] = configs
        self.results: Dict[str, List[TestResult]] = {
            config.name: [] for config in configs}

    def generate_test_data(self) -> List[Dict[str, Any]]:
        """
        Generate test data for the phone price prediction model.

        Returns:
            Dict[str, Any]: Test data matching the model's input format
        """
        return [{
        'screen_size': 6.1,
        'rear_camera_mp': 12,
        'front_camera_mp': 12,
        'internal_memory': 128,
        'ram': 4,
        'battery': 3000,
        'weight': 175,
        'days_used': 30,
        'device_brand': 'OnePlus',
        'os': 'Android',
        '4g': 'yes',
        '5g': 'no'
    }]

    def single_request(self, config: TestConfig) -> TestResult:
        """
        Make a single request to a deployment endpoint.

        Args:
            config (TestConfig): Configuration for the deployment to test

        Returns:
            TestResult: Contains deployment type, latency, status code, and response
        """
        data: Dict[str, Any] = self.generate_test_data()
        start_time: float = time.time()

        try:
            response: requests.Response = requests.post(
                config.endpoint_url,
                json=data,
                headers=config.headers if config.headers else {},
                timeout=30
            )
            latency: float = time.time() - start_time
            status_code: int = response.status_code

            if status_code == 200:
                result: Dict[str, Any] = response.json()
            else:
                result = None

            return TestResult(
                deployment_type=config.deployment_type,
                latency=latency,
                status_code=status_code,
                response=result,
                timestamp=start_time
            )

        except Exception as e:
            return TestResult(
                deployment_type=config.deployment_type,
                latency=time.time() - start_time,
                status_code=500,
                response={"error": str(e)},
                timestamp=start_time
            )

    def concurrent_test(self,
                        config: TestConfig,
                        num_requests: int,
                        max_workers: int) -> List[TestResult]:
        """
        Run concurrent requests for a specific deployment.

        Args:
            config (TestConfig): Configuration for the deployment to test
            num_requests (int): Number of requests to make
            max_workers (int): Maximum number of concurrent workers

        Returns:
            List[TestResult]: List of test results
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures: List[concurrent.futures.Future] = [
                executor.submit(self.single_request, config)
                for _ in range(num_requests)
            ]
            results: List[TestResult] = [
                future.result()
                for future in concurrent.futures.as_completed(futures)
            ]
        return results

    def run_performance_test(self,
                             num_requests: int = 100,
                             max_workers: int = 10,
                             test_rounds: int = 3) -> None:
        """
        Run complete performance test for all deployments with multiple rounds.

        Args:
            num_requests (int): Requests per round
            max_workers (int): Concurrent workers
            test_rounds (int): Number of test rounds
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for config in self.configs:
            print(f"\nTesting {config.name} deployment:")
            print(f"Starting performance test: {num_requests} requests, "
                  f"{max_workers} concurrent workers, {test_rounds} rounds")

            for round in range(test_rounds):
                print(f"Test round {round + 1}/{test_rounds}")
                round_results: List[TestResult] = self.concurrent_test(
                    config, num_requests, max_workers
                )
                self.results[config.name].extend(round_results)
                time.sleep(2)  # Cool-down period between rounds

        # self.analyze_results(timestamp)

    def analyze_results(self, timestamp: str) -> None:
        """
        Analyze and compare results from different deployments.

        Args:
            timestamp (str): Timestamp for saving results
        """
        print("\nPerformance Test Results Comparison:")

        # Prepare data for plotting
        plot_data = {}
        summary_data = []

        for config_name, results in self.results.items():
            latencies: List[float] = [r.latency for r in results]
            status_codes: List[int] = [r.status_code for r in results]

            # Calculate statistics
            success_rate: float = status_codes.count(
                200) / len(status_codes) * 100
            avg_latency: float = np.mean(latencies)
            p50_latency: float = np.percentile(latencies, 50)
            p95_latency: float = np.percentile(latencies, 95)
            p99_latency: float = np.percentile(latencies, 99)

            plot_data[config_name] = latencies

            print(f"\n{config_name} Results:")
            print(f"Total Requests: {len(results)}")
            print(f"Success Rate: {success_rate:.2f}%")
            print(f"Average Latency: {avg_latency*1000:.2f}ms")
            print(f"P50 Latency: {p50_latency*1000:.2f}ms")
            print(f"P95 Latency: {p95_latency*1000:.2f}ms")
            print(f"P99 Latency: {p99_latency*1000:.2f}ms")

            # Save detailed results
            df = pd.DataFrame([
                {
                    'deployment_type': r.deployment_type.value,
                    'latency': r.latency,
                    'status_code': r.status_code,
                    'response': str(r.response),
                    'timestamp': r.timestamp
                }
                for r in results
            ])
            df.to_csv(f'performance_test_results_{config_name}_{
                      timestamp}.csv', index=False)

            summary_data.append({
                'deployment_type': config_name,
                'success_rate': success_rate,
                'avg_latency': avg_latency,
                'p50_latency': p50_latency,
                'p95_latency': p95_latency,
                'p99_latency': p99_latency
            })

        # Save summary comparison
        pd.DataFrame(summary_data).to_csv(
            f'performance_comparison_summary_{timestamp}.csv', index=False)

        # Create comparison plots
        self.create_comparison_plots(plot_data, timestamp)

    def create_comparison_plots(self, plot_data: Dict[str, List[float]], timestamp: str) -> None:
        """
        Create comparison plots for the different deployments.

        Args:
            plot_data (Dict[str, List[float]]): Latency data for each deployment
            timestamp (str): Timestamp for saving plots
        """
        plt.figure(figsize=(12, 6))
        plt.boxplot(plot_data.values(), labels=plot_data.keys())
        plt.title('Latency Distribution Comparison')
        plt.ylabel('Latency (seconds)')
        plt.yscale('log')
        plt.grid(True)
        plt.savefig(f'latency_comparison_boxplot_{timestamp}.png')
        plt.close()

        # Create histogram
        plt.figure(figsize=(12, 6))
        for name, latencies in plot_data.items():
            plt.hist(latencies, alpha=0.5, label=name, bins=50)
        plt.title('Latency Distribution Histogram')
        plt.xlabel('Latency (seconds)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'latency_comparison_histogram_{timestamp}.png')
        plt.close()


def load_config() -> Dict[str, Any]:
    """
    Load test configuration from YAML file.

    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    config_path = Path(__file__).parent / "config" / "test_config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_directories() -> Dict[str, Path]:
    """
    Create necessary directories for storing results.

    Returns:
        Dict[str, Path]: Dictionary containing paths to result directories
    """
    base_dir = Path(__file__).parent.parent.parent / "results"

    dirs = {
        'raw': base_dir / 'raw',
        'plots': base_dir / 'plots',
        'summaries': base_dir / 'summaries'
    }

    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    return dirs


def main() -> None:
    # Load configuration
    config = load_config()

    # Setup directories
    dirs = setup_directories()

    # Create test configurations
    configs = [
        TestConfig(
            name=dep_config['name'],
            endpoint_url=dep_config['endpoint_url'],
            deployment_type=DeploymentType[dep_name.upper()],
            headers=dep_config.get('headers', {})
        )
        for dep_name, dep_config in config['deployments'].items()
    ]

    tester: ModelPerformanceTester = ModelPerformanceTester(configs)

    # Run performance test
    test_params = config['test_parameters']
    tester.run_performance_test(
        num_requests=test_params['requests_per_round'],
        max_workers=test_params['max_workers'],
        test_rounds=test_params['test_rounds']
    )


if __name__ == "__main__":
    main()
