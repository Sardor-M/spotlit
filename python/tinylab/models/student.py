"""The student network. MLX, Apple Silicon. See ADR-0006."""

from __future__ import annotations

from dataclasses import dataclass, asdict

import mlx.core as mx
import mlx.nn as nn

EPS = 1e-8


@dataclass(frozen=True, slots=True)
class StudentConfig:
    num_buckets: int = 4096
    d_table: int = 64
    hidden: int = 128
    d_out: int = 128

    @property
    def parameters(self) -> int:
        table = self.num_buckets * self.d_table
        mlp = (
            self.d_table * self.hidden + self.hidden
            + self.hidden * self.hidden + self.hidden
            + self.hidden * self.d_out + self.d_out
        )
        return table + mlp

    def describe(self) -> str:
        return (
            f"buckets={self.num_buckets} d_table={self.d_table} "
            f"hidden={self.hidden} d_out={self.d_out} params={self.parameters:,}"
        )

    def to_dict(self) -> dict:
        return asdict(self)


CONFIG_A = StudentConfig(num_buckets=4096, d_table=64)
CONFIG_B = StudentConfig(num_buckets=16384, d_table=16)
CONFIGS = {"A": CONFIG_A, "B": CONFIG_B}


class Student(nn.Module):
    """Hashed n-gram table, mean-pooled, through a small MLP to a normalized vector."""

    def __init__(self, config: StudentConfig = CONFIG_A):
        super().__init__()
        self.config = config
        self.table = nn.Embedding(config.num_buckets, config.d_table)
        self.fc1 = nn.Linear(config.d_table, config.hidden)
        self.fc2 = nn.Linear(config.hidden, config.hidden)
        self.fc3 = nn.Linear(config.hidden, config.d_out)

    def __call__(self, buckets: mx.array, counts: mx.array, mask: mx.array) -> mx.array:
        weights = (counts * mask)[..., None]
        pooled = (self.table(buckets) * weights).sum(axis=1)
        pooled = pooled / mx.maximum(mask.sum(axis=1, keepdims=True), 1.0)

        hidden = nn.relu(self.fc1(pooled))
        hidden = nn.relu(self.fc2(hidden))
        out = self.fc3(hidden)
        return out / mx.maximum(mx.linalg.norm(out, axis=-1, keepdims=True), EPS)


def relational_loss(student: mx.array, teacher: mx.array) -> mx.array:
    """Match the teacher's pairwise similarity structure, excluding the diagonal."""
    student_similarity = student @ student.T
    teacher_similarity = teacher @ teacher.T

    n = student.shape[0]
    off_diagonal = 1.0 - mx.eye(n)
    squared = (student_similarity - teacher_similarity) ** 2 * off_diagonal
    return squared.sum() / mx.maximum(off_diagonal.sum(), 1.0)


def listwise_loss(student: mx.array, teacher: mx.array, temperature: float = 0.05) -> mx.array:
    """KL from the teacher's softmax over each row's neighbours to the student's."""
    n = student.shape[0]
    diagonal = mx.eye(n) * 1e4
    teacher_logits = (teacher @ teacher.T) / temperature - diagonal
    student_logits = (student @ student.T) / temperature - diagonal
    target = mx.softmax(teacher_logits, axis=-1)
    log_target = teacher_logits - mx.logsumexp(teacher_logits, axis=-1, keepdims=True)
    log_student = student_logits - mx.logsumexp(student_logits, axis=-1, keepdims=True)
    return (target * (log_target - log_student)).sum(axis=-1).mean()


LOSSES = {"mse": relational_loss, "listwise": listwise_loss}
