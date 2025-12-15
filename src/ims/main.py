import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Optional
from pathlib import Path
import sys
import os
 
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
 
from utils import characterise
from utils.measures import ExecutionTime
from utils.methods import MatrixInverse, LU, Cholesky, QR, SparseLU
 
def generate_general_dense(n: int) -> Tuple[np.ndarray, np.ndarray]:
    A = np.random.randn(n, n) + 5 * np.eye(n)
    b = np.random.randn(n)
    return A, b
 
def generate_symmetric(n: int) -> Tuple[np.ndarray, np.ndarray]:
    eps = 1
    x0, x1 = 1.0, 10.0
    blim = 15.0
 
    lambdas = np.random.uniform(x0, x1, n)
    V = np.random.uniform(0, 1, (n, n)) + eps * np.identity(n)
    V = np.tril(V)
    A = V @ V.T + np.diag(lambdas)
    b = np.random.uniform(-blim, blim, n)
    return A, b
 
def generate_tridiagonal(n: int) -> Tuple[np.ndarray, np.ndarray]:
    A = np.zeros((n, n))
    np.fill_diagonal(A, 2)
    np.fill_diagonal(A[1:], -1)
    np.fill_diagonal(A[:, 1:], -1)
    b = np.random.randn(n)
    return A, b
 
def generate_sparse(dimension: int, rho: float, a: float = 5.0, delta: float = 0.01, b_density: float = None) -> Tuple[np.ndarray, np.ndarray]:
    def sample_offdiagonal(n, num_nonzero):
        rows, cols = np.where(~np.identity(n, dtype=bool))
        idx = np.random.choice(rows.size, size=num_nonzero, replace=False)
        return rows[idx], cols[idx]
 
    n = dimension
 
    if rho < 1 / n:
        raise ValueError(
            f"rho must be at least 1/n to ensure invertibility. In this case, rho = {rho}, 1/n = {1 / n}"
        )
 
    A = np.zeros((n, n))
 
    num_nonzero = int(rho * n * n - n)
 
    if num_nonzero > 0:
        rows, cols = sample_offdiagonal(n, num_nonzero)
        values = np.random.uniform(-a, a, size=num_nonzero)
        A[rows, cols] = values
 
    rowSums = np.sum(np.abs(A), axis=1)
    eps = np.random.uniform(delta, 1, size=n)
    diagonal_values = np.diag(rowSums + eps)
    A += diagonal_values
 
    if b_density is None:
        num_nonzero_b = int(rho * n)
    else:
        if b_density < 0 or b_density > 1:
            print(f"b_density must be between 0 and 1, got {b_density}")
        num_nonzero_b = int(b_density * n)
 
    indices_b = np.random.choice(n, size=num_nonzero_b, replace=False)
    b = np.zeros(n)
    b[indices_b] = np.random.uniform(-a, a, size=num_nonzero_b)
 
    return A, b
 
 
def generate_gershgorin_matrix(n: int, target_density: float) -> Tuple[np.ndarray, np.ndarray]:
    M = np.zeros((n, n))
 
    total_elements = n * n
    num_nonzero = int(target_density * total_elements)
 
    used = set()
    rows, cols = [], []
 
    max_attempts = num_nonzero * 10
    attempts = 0
 
    while len(rows) < (num_nonzero - n) and attempts < max_attempts:
        r = np.random.randint(0, n)
        c = np.random.randint(0, n)
 
        if r != c and (r, c) not in used:
            used.add((r, c))
            rows.append(r)
            cols.append(c)
 
        attempts += 1
 
    if rows:
        values = np.random.uniform(-10, 10, len(rows))
        M[rows, cols] = values
 
    row_sums = np.sum(np.abs(M), axis=1)
    δ = 0.01
    ε = np.random.uniform(δ, 1, n)
    diagonal_values = row_sums + ε
 
    D = np.diag(diagonal_values)
    A = M + D
 
    b = np.random.randn(n)
 
    return A, b
 
def compute_matrix_stats(A: np.ndarray, b: np.ndarray, method_name: str, x_solution: np.ndarray) -> dict:
    stats = {
        'Sparsity': 1.0 - (np.count_nonzero(A) / A.size),
        'Density': np.count_nonzero(A) / A.size
    }
    return stats
 
@characterise(
    methods=[MatrixInverse, LU, QR],
    measures=[ExecutionTime],
    realisations=5,
    iterations=100
)
def experiment_1_dense(n: int):
    return generate_general_dense(n)
 
@characterise(
    methods=[MatrixInverse, LU, Cholesky, QR],
    measures=[ExecutionTime],
    realisations=5,
    iterations=100
)
def experiment_2_symmetric(n: int):
    return generate_symmetric(n)
 
@characterise(
    methods=[SparseLU, LU, QR, MatrixInverse],
    measures=[ExecutionTime],
    realisations=5,
    iterations=100
)
def experiment_3_sparse_dense_b(n: int, density: float):
    return generate_sparse(n, density, b_density=1.0)
 
@characterise(
    methods=[SparseLU, LU, QR],
    measures=[ExecutionTime],
    realisations=5,
    iterations=100
)
def experiment_4_sparse_both(n: int, density: float):
    return generate_sparse(n, density, b_density=density)
 
@characterise(
    methods=[SparseLU, LU, Cholesky, MatrixInverse, QR],
    measures=[ExecutionTime],
    realisations=5,
    iterations=100
)
def experiment_5_trid_and_gershgorin_matrices_dense(n: int, matrix_type: str):
    if matrix_type == 'tridiagonal':
        return generate_tridiagonal(n)
    elif matrix_type == 'gershgorin':
        density = (3 * n - 2) / (n ** 2)
        return generate_gershgorin_matrix(n, density)
 
@characterise(
    methods=[SparseLU, LU, Cholesky],
    measures=[ExecutionTime],
    realisations=5,
    iterations=100
)
def experiment_6_trid_and_gershgorin_matrices_sparse(n: int, matrix_type: str):
    if matrix_type == 'tridiagonal':
        A, _ = generate_tridiagonal(n)
    elif matrix_type == 'gershgorin':
        density = (3 * n - 2) / (n ** 2)
        A, _ = generate_gershgorin_matrix(n, density)
 
    b_density = (3 * n - 2) / (n ** 2)
    num_nonzero_b = int(b_density * n)
    indices = np.random.choice(n, size=num_nonzero_b, replace=False)
    b = np.zeros(n)
    b[indices] = np.random.randn(num_nonzero_b)
 
    return A, b
 
def run_experiment_1_2():
    print("Experiments 1 and 2 for dense matrices")
 
    dimensions = [50, 100, 200, 400]
    all_results = []
 
    for d in dimensions:
        print(f"Dimension {d}")
 
        res1 = experiment_1_dense(d)
        res1 = res1.with_columns([
            pl.lit(d).alias("Dimension"),
            pl.lit("General dense").alias("MatrixType"),
            pl.lit("Exp1: Dense").alias("Experiment"),
            pl.lit(1.0).alias("Density")
        ])
 
        res2 = experiment_2_symmetric(d)
        res2 = res2.with_columns([
            pl.lit(d).alias("Dimension"),
            pl.lit("SPD").alias("MatrixType"),
            pl.lit("Exp2: SPD").alias("Experiment"),
            pl.lit(1.0).alias("Density")
        ])
 
        all_results.extend([res1, res2])
 
    return pl.concat(all_results)
 
 
def run_experiment_3():
    print("Experiment 3 for sparse matrices (density b)")
 
    densities = [0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 1.0]
    fixed_size = 200
 
    all_results = []
 
    for rho in densities:
        print(f"Density {rho:.3f}")
        res = experiment_3_sparse_dense_b(fixed_size, rho)
        res = res.with_columns([
            pl.lit(fixed_size).alias("Dimension"),
            pl.lit("Sparse").alias("MatrixType"),
            pl.lit("Exp3: Sparse A, Dense b").alias("Experiment"),
            pl.lit(rho).alias("Density"),
        ])
        all_results.append(res)
 
    return pl.concat(all_results)
 
def run_experiment_4():
    print("Experiment 4 for sparse matrices and vectors")
 
    densities = [0.01, 0.05, 0.1, 0.2, 0.5]
    fixed_size = 200
 
    all_results = []
 
    for rho in densities:
        print(f"Density {rho:.3f}")
        res = experiment_4_sparse_both(fixed_size, rho)
        res = res.with_columns([
            pl.lit(fixed_size).alias("Dimension"),
            pl.lit("Sparse").alias("MatrixType"),
            pl.lit("Exp4: Sparse A, Sparse b").alias("Experiment"),
            pl.lit(rho).alias("Density"),
        ])
        all_results.append(res)
 
    return pl.concat(all_results)
 
 
def run_experiment_5():
    print("Experiment 5 for dense triadiagonal/Gershgorin matrices comparison")
 
    dimensions = [100, 200, 400, 800]
    all_results = []
 
    for n in dimensions:
        print(f"Dimension {n}")
 
        res_tri = experiment_5_trid_and_gershgorin_matrices_dense(n, 'tridiagonal')
        density_tri = (3 * n - 2) / (n ** 2)
        res_tri = res_tri.with_columns([
            pl.lit(n).alias("Dimension"),
            pl.lit(density_tri).alias("Density"),
            pl.lit("Tridiagonal").alias("MatrixType"),
            pl.lit("Exp5: dense b)").alias("Experiment")
        ])
 
        res_gersh = experiment_5_trid_and_gershgorin_matrices_dense(n, 'gershgorin')
        res_gersh = res_gersh.with_columns([
            pl.lit(n).alias("Dimension"),
            pl.lit(density_tri).alias("Density"),
            pl.lit("Gershgorin").alias("MatrixType"),
            pl.lit("Exp5 dense b").alias("Experiment")
        ])
 
        all_results.extend([res_tri, res_gersh])
 
    return pl.concat(all_results)
 
 
def run_experiment_6():
    print("Experiment 6 for sparse triadiagonal/Gershgorin matrices comparison")
 
    dimensions = [100, 200, 400, 800]
    all_results = []
 
    for n in dimensions:
        print(f"Dimension {n}")
 
        res_tri = experiment_6_trid_and_gershgorin_matrices_sparse(n, 'tridiagonal')
        density_tri = (3 * n - 2) / (n ** 2)
        res_tri = res_tri.with_columns([
            pl.lit(n).alias("Dimension"),
            pl.lit(density_tri).alias("Density"),
            pl.lit("Tridiagonal").alias("MatrixType"),
            pl.lit("Exp6 sparse b").alias("Experiment")
        ])
 
        res_gersh = experiment_6_trid_and_gershgorin_matrices_sparse(n, 'gershgorin')
        res_gersh = res_gersh.with_columns([
            pl.lit(n).alias("Dimension"),
            pl.lit(density_tri).alias("Density"),
            pl.lit("Gershgorin").alias("MatrixType"),
            pl.lit("Exp6 sparse b").alias("Experiment")
        ])
 
        all_results.extend([res_tri, res_gersh])
 
    return pl.concat(all_results)
 
 
def create_plots(results: pl.DataFrame, save_dir: str = "plots"):
    Path(save_dir).mkdir(exist_ok=True)
 
    exp1_2 = results.filter(
        (pl.col("Experiment") == "Exp1 Dense") |
        (pl.col("Experiment") == "Exp2 SPD")
    )
 
    if len(exp1_2) > 0:
        plt.figure(figsize=(10, 6))
 
        for exp_name in exp1_2["Experiment"].unique():
            exp_data = exp1_2.filter(pl.col("Experiment") == exp_name)
 
            for method in exp_data["Method"].unique():
                method_data = exp_data.filter(pl.col("Method") == method)
 
                grouped = method_data.group_by("Dimension").agg(
                    pl.col("ExecutionTime").mean().alias("MeanTime"),
                    pl.col("ExecutionTime").std().alias("StdTime")
                ).sort("Dimension")
 
                if len(grouped) > 0:
                    plt.errorbar(
                        grouped["Dimension"].to_numpy(),
                        grouped["MeanTime"].to_numpy(),
                        yerr=grouped["StdTime"].to_numpy(),
                        label=f"{exp_name} - {method}",
                        capsize=5,
                        marker='o',
                        alpha=0.7
                    )
 
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Matrix dimension')
        plt.ylabel('Execution time (s)')
        plt.title('Experiments 1 & 2 for dense matrices')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/exp1_2_time_vs_size.pdf', format='pdf', dpi=300)
        plt.show()
 
    exp3 = results.filter(pl.col("Experiment") == "Exp3: Sparse A, dense b")
 
    if len(exp3) > 0:
        plt.figure(figsize=(10, 6))
 
        for method in exp3["Method"].unique():
            method_data = exp3.filter(pl.col("Method") == method)
 
            grouped = method_data.group_by("Density").agg(
                pl.col("ExecutionTime").mean().alias("MeanTime"),
                pl.col("ExecutionTime").std().alias("StdTime")
            ).sort("Density")
 
            if len(grouped) > 0:
                plt.errorbar(
                    grouped["Density"].to_numpy(),
                    grouped["MeanTime"].to_numpy(),
                    yerr=grouped["StdTime"].to_numpy(),
                    label=str(method),
                    capsize=5,
                    marker='s',
                    alpha=0.7
                )
 
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Matrix density')
        plt.ylabel('Execution time (s)')
        plt.title('Experiment 3 for sparse matrices (density b)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{save_dir}/exp3_time_vs_density.pdf', format='pdf', dpi=300)
        plt.show()
 
    exp4 = results.filter(pl.col("Experiment") == "Exp4: Sparse A, Sparse b")
 
    if len(exp4) > 0 and len(exp3) > 0:
        plt.figure(figsize=(12, 6))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
 
        for method in exp3["Method"].unique():
            if str(method) != "MatrixInverse":
                method_data = exp3.filter(pl.col("Method") == method)
                grouped = method_data.group_by("Density").agg(
                    pl.col("ExecutionTime").mean().alias("MeanTime")
                ).sort("Density")
 
                ax1.plot(grouped["Density"], grouped["MeanTime"],
                         'o-', label=str(method), alpha=0.7)
 
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlabel('Matrix density')
        ax1.set_ylabel('Execution time (s)')
        ax1.set_title('Sparse A, dense b')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
 
        for method in exp4["Method"].unique():
            method_data = exp4.filter(pl.col("Method") == method)
            grouped = method_data.group_by("Density").agg(
                pl.col("ExecutionTime").mean().alias("MeanTime")
            ).sort("Density")
 
            ax2.plot(grouped["Density"], grouped["MeanTime"],
                     's-', label=str(method), alpha=0.7)
 
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xlabel('Matrix density')
        ax2.set_ylabel('Execution time (s)')
        ax2.set_title('Sparse A, sparse b')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
 
        plt.tight_layout()
        plt.savefig(f'{save_dir}/exp3_4_comparison.pdf', format='pdf', dpi=300)
        plt.show()
 
    exp5 = results.filter(pl.col("Experiment") == "Exp5: Special (dense b)")
 
    if len(exp5) > 0:
        plt.figure(figsize=(10, 6))
 
        for matrix_type in exp5["MatrixType"].unique():
            for method in exp5["Method"].unique():
                subset = exp5.filter(
                    (pl.col("MatrixType") == matrix_type) &
                    (pl.col("Method") == method)
                )
 
                grouped = subset.group_by("Dimension").agg(
                    pl.col("ExecutionTime").mean().alias("MeanTime"),
                    pl.col("ExecutionTime").std().alias("StdTime")
                ).sort("Dimension")
 
                if len(grouped) > 0:
                    plt.errorbar(
                        grouped["Dimension"].to_numpy(),
                        grouped["MeanTime"].to_numpy(),
                        yerr=grouped["StdTime"].to_numpy(),
                        label=f"{matrix_type} - {method}",
                        capsize=3,
                        marker='o' if matrix_type == "Tridiagonal" else 's',
                        alpha=0.7
                    )
 
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Matrix dimension')
        plt.ylabel('Execution time (s)')
        plt.title('Experiment 5 for dense triadiagonal/Gershgorin matrices comparison')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/exp5_special_matrices.pdf', format='pdf', dpi=300)
        plt.show()
 
    spd_results = results.filter(
        (pl.col("MatrixType") == "SPD") &
        (pl.col("Dimension") >= 100)
    )
 
 
def compute_stats(results: pl.DataFrame) -> pl.DataFrame:
    stats = results.group_by(["Experiment", "Method", "Dimension"]).agg([
        pl.col("ExecutionTime").mean().alias("MeanTime"),
        pl.col("ExecutionTime").std().alias("StdTime"),
        pl.col("ExecutionTime").min().alias("MinTime"),
        pl.col("ExecutionTime").max().alias("MaxTime"),
        pl.count().alias("Samples")
    ])
 
    return stats
 
def print_summary_statistics(results: pl.DataFrame):
    for exp in sorted(results["Experiment"].unique()):
        print(f"\n{exp}:")
        exp_data = results.filter(pl.col("Experiment") == exp)
 
        stats = exp_data.group_by("Method").agg([
            pl.col("ExecutionTime").mean().alias("MeanTime"),
            pl.col("ExecutionTime").std().alias("StdTime"),
            pl.col("Dimension").mean().alias("AvgDimension"),
            pl.count().alias("Samples")
        ])
 
        for row in stats.iter_rows(named=True):
            method_name = str(row['Method'])
            mean_time = row['MeanTime']
            std_time = row['StdTime']
 
            print(f"  {method_name:15s}: {mean_time:.6f} ± {std_time:.6f} s "
                  f"(n={row['Samples']}, avg size={row['AvgDimension']:.0f})")
 
def main(run_full: bool = False):
    Path("results").mkdir(exist_ok=True)
    Path("plots").mkdir(exist_ok=True)
 
    all_results = []
 
    if run_full:
        experiments = [
            ("1 & 2", run_experiment_1_2),
            ("3", run_experiment_3),
            ("4", run_experiment_4),
            #("5", run_experiment_5),
            #("6", run_experiment_6)
        ]
    else:
        experiments = [
            ("1 & 2", run_experiment_1_2),
            #("5", run_experiment_5)
        ]
 
    for exp_name, exp_func in experiments:
        results = exp_func()
        all_results.append(results)
        print(f"Completed successfully ({len(results)} records)")
 
    combined_results = pl.concat(all_results, how="vertical")
 
    stats = compute_stats(combined_results)
    print_summary_statistics(combined_results)
 
    print("\nCreating plots")
    create_plots(combined_results)
    print("Analysis completed")
    print(f"Total records collected: {len(combined_results)}")
 
    return combined_results, stats
 
if __name__ == "__main__":
    results, stats = main(run_full=True)
