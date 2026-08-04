from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .models import RoofCandidate


@dataclass(frozen=True, slots=True)
class _ScoredContour:
    contour: np.ndarray
    mask: np.ndarray
    score: float
    label: str
    centroid: tuple[int, int]


class MultiMaterialRoofDetector:
    """Explainable image-only roof detector for high-resolution aerial imagery.

    It creates candidate regions for warm tiled, neutral/metal and green roof
    surfaces, then filters them using size, solidity, rectangularity and edge
    evidence. It is a practical baseline, not a universal trained detector.
    """

    def __init__(self, min_area_px: float = 450.0) -> None:
        self.min_area_px = min_area_px

    def detect(self, image: np.ndarray, max_buildings: int) -> list[RoofCandidate]:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        masks = {
            "warm_surface": self._warm_mask(hsv),
            "neutral_surface": self._neutral_mask(image, hsv),
            "green_surface": self._green_mask(image, hsv),
        }

        all_candidates: list[_ScoredContour] = []
        for label, mask in masks.items():
            all_candidates.extend(self._extract_candidates(image, mask, label))

        all_candidates.sort(key=lambda item: item.score, reverse=True)
        selected: list[_ScoredContour] = []
        for candidate in all_candidates:
            if any(
                self._intersection_over_union(candidate.mask, existing.mask) > 0.35
                for existing in selected
            ):
                continue
            selected.append(candidate)
            if len(selected) >= max_buildings:
                break

        selected.sort(key=lambda item: (item.centroid[1], item.centroid[0]))
        return [
            RoofCandidate(
                contour=item.contour,
                mask=item.mask,
                detection_score=item.score,
                detector_label=item.label,
                centroid_px=item.centroid,
            )
            for item in selected
        ]

    @staticmethod
    def _warm_mask(hsv: np.ndarray) -> np.ndarray:
        first = cv2.inRange(
            hsv,
            np.array([0, 38, 45], dtype=np.uint8),
            np.array([30, 255, 255], dtype=np.uint8),
        )
        second = cv2.inRange(
            hsv,
            np.array([165, 35, 45], dtype=np.uint8),
            np.array([179, 255, 255], dtype=np.uint8),
        )
        return MultiMaterialRoofDetector._clean_mask(first | second, close_size=7)

    @staticmethod
    def _neutral_mask(image: np.ndarray, hsv: np.ndarray) -> np.ndarray:
        low_saturation = cv2.inRange(
            hsv,
            np.array([0, 0, 72], dtype=np.uint8),
            np.array([179, 62, 235], dtype=np.uint8),
        )
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 55, 145)
        edge_support = cv2.dilate(
            edges,
            cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
        )
        mask = cv2.bitwise_and(low_saturation, edge_support)
        return MultiMaterialRoofDetector._clean_mask(mask, close_size=11)

    @staticmethod
    def _green_mask(image: np.ndarray, hsv: np.ndarray) -> np.ndarray:
        green = cv2.inRange(
            hsv,
            np.array([34, 35, 35], dtype=np.uint8),
            np.array([92, 210, 225], dtype=np.uint8),
        )
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        local_mean = cv2.blur(gray, (9, 9))
        deviation = cv2.absdiff(gray, local_mean)
        smooth = cv2.threshold(deviation, 22, 255, cv2.THRESH_BINARY_INV)[1]
        mask = cv2.bitwise_and(green, smooth)
        return MultiMaterialRoofDetector._clean_mask(mask, close_size=13)

    @staticmethod
    def _clean_mask(mask: np.ndarray, close_size: int) -> np.ndarray:
        opened = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        return cv2.morphologyEx(
            opened,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (close_size, close_size)
            ),
        )

    def _extract_candidates(
        self, image: np.ndarray, mask: np.ndarray, label: str
    ) -> list[_ScoredContour]:
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        image_area = float(image.shape[0] * image.shape[1])
        candidates: list[_ScoredContour] = []

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edge_map = cv2.Canny(gray, 45, 135)

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.min_area_px or area > image_area * 0.18:
                continue
            x, y, width, height = cv2.boundingRect(contour)
            if min(width, height) < 12:
                continue
            if x <= 1 or y <= 1 or x + width >= image.shape[1] - 1:
                continue

            hull = cv2.convexHull(contour)
            hull_area = float(cv2.contourArea(hull))
            rectangle_area = float(width * height)
            solidity = area / hull_area if hull_area else 0.0
            rectangularity = area / rectangle_area if rectangle_area else 0.0
            if solidity < 0.55 or rectangularity < 0.18:
                continue

            candidate_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.drawContours(candidate_mask, [contour], -1, 255, thickness=-1)
            edge_density = float(np.mean(edge_map[candidate_mask == 255] > 0))
            if label == "neutral_surface" and edge_density < 0.025:
                continue

            perimeter = cv2.arcLength(contour, True)
            compactness = (
                min(1.0, 4.0 * np.pi * area / (perimeter * perimeter))
                if perimeter > 0
                else 0.0
            )
            size_score = min(1.0, area / max(2500.0, image_area * 0.008))
            score = (
                0.30 * solidity
                + 0.25 * min(1.0, rectangularity / 0.70)
                + 0.20 * size_score
                + 0.15 * min(1.0, edge_density / 0.10)
                + 0.10 * compactness
            )
            if label == "warm_surface":
                score += 0.08
            if label == "green_surface":
                score -= 0.05

            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue
            centroid = (
                int(moments["m10"] / moments["m00"]),
                int(moments["m01"] / moments["m00"]),
            )
            candidates.append(
                _ScoredContour(
                    contour=contour,
                    mask=candidate_mask,
                    score=float(np.clip(score, 0.0, 1.0)),
                    label=label,
                    centroid=centroid,
                )
            )
        return candidates

    @staticmethod
    def _intersection_over_union(first: np.ndarray, second: np.ndarray) -> float:
        intersection = np.count_nonzero((first > 0) & (second > 0))
        union = np.count_nonzero((first > 0) | (second > 0))
        return float(intersection / union) if union else 0.0
