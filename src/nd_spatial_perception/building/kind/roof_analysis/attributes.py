from __future__ import annotations

from collections import Counter

import cv2
import numpy as np

from .models import GeoImage, RoofCandidate, RoofResult


class RoofAttributeExtractor:
    """Extracts conservative roof attributes and per-attribute confidence."""

    def extract(
        self,
        geo_image: GeoImage,
        candidate: RoofCandidate,
        index: int,
    ) -> RoofResult:
        contour = candidate.contour
        polygon_px = self._simplify_contour(contour)
        polygon_lonlat = geo_image.georeferencer.pixel_to_lonlat(polygon_px)
        area_m2 = geo_image.georeferencer.area_m2(polygon_px)

        material, material_confidence = self._classify_material(
            geo_image.bgr, candidate.mask
        )
        roof_type, type_confidence, ridge_angle = self._classify_type(
            geo_image.bgr, candidate.mask, contour
        )
        orientation_deg = (
            ridge_angle
            if ridge_angle is not None
            else self._contour_orientation(contour)
        )
        orientation = self._orientation_name(orientation_deg)
        orientation_confidence = self._orientation_confidence(
            contour, ridge_angle is not None
        )

        panel_count, panel_confidence = self._detect_solar_panels(
            geo_image.bgr, candidate.mask
        )
        superstructure_count, superstructure_confidence = self._detect_superstructures(
            geo_image.bgr, candidate.mask
        )
        condition, condition_confidence = self._visible_condition(
            geo_image.bgr, candidate.mask
        )

        polygon_confidence = self._polygon_confidence(contour, candidate.detection_score)
        area_confidence = min(
            polygon_confidence,
            polygon_confidence * geo_image.georeferencer.area_confidence_factor,
        )

        notes = self._build_notes(
            georeference_quality=geo_image.georeferencer.quality,
            roof_type=roof_type,
            material=material,
        )

        return RoofResult(
            building_id=f"building_{index:03d}",
            source_used=geo_image.source_description,
            polygon_lonlat=polygon_lonlat,
            polygon_px=polygon_px,
            area_m2=area_m2,
            roof_type=roof_type,
            material=material,
            orientation=orientation,
            orientation_deg=orientation_deg,
            slope_deg=None,
            solar_panels_present=panel_count > 0,
            solar_panel_count=panel_count,
            superstructure_count=superstructure_count,
            visible_condition=condition,
            confidence={
                "polygon": polygon_confidence,
                "area": area_confidence,
                "type": type_confidence,
                "material": material_confidence,
                "orientation": orientation_confidence,
                "slope": 0.0,
                "solar_panels": panel_confidence,
                "superstructures": superstructure_confidence,
                "visible_condition": condition_confidence,
            },
            notes=notes,
        )

    @staticmethod
    def _simplify_contour(contour: np.ndarray) -> list[tuple[float, float]]:
        hull = cv2.convexHull(contour)
        perimeter = cv2.arcLength(hull, True)
        simplified = cv2.approxPolyDP(hull, 0.012 * perimeter, True)
        points = [
            (float(point[0][0]), float(point[0][1]))
            for point in simplified
        ]
        if points and points[0] != points[-1]:
            points.append(points[0])
        return points

    @staticmethod
    def _classify_material(
        image: np.ndarray, mask: np.ndarray
    ) -> tuple[str, float]:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        pixels = hsv[mask > 0]
        gray_pixels = gray[mask > 0]
        if pixels.size == 0:
            return "unknown", 0.0

        hue = float(np.median(pixels[:, 0]))
        saturation = float(np.median(pixels[:, 1]))
        value = float(np.median(pixels[:, 2]))
        texture = float(np.std(gray_pixels))

        if (hue <= 30 or hue >= 165) and saturation >= 45:
            confidence = min(0.88, 0.52 + saturation / 700.0 + texture / 300.0)
            return "tiled", confidence
        if 34 <= hue <= 92 and saturation >= 40 and texture < 42:
            return "green_roof", min(0.82, 0.52 + saturation / 800.0)
        if saturation < 48 and value > 175 and texture < 34:
            return "metal", 0.62
        if saturation < 42 and value > 185 and texture >= 34:
            return "glass", 0.48
        if saturation < 65 and 75 <= value <= 190:
            return "flat/gravel", 0.53
        return "unknown", 0.25

    def _classify_type(
            self,
            image: np.ndarray,
            mask: np.ndarray,
            contour: np.ndarray,
    ) -> tuple[str, float, float | None]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 55, 150)
        masked_edges = cv2.bitwise_and(edges, edges, mask=mask)

        lines = cv2.HoughLinesP(
            masked_edges,
            1,
            np.pi / 180.0,
            threshold=22,
            minLineLength=18,
            maxLineGap=8,
        )

        line_angles: list[float] = []

        if lines is not None:
            normalized_lines = np.asarray(lines).reshape(-1, 4)

            for x1, y1, x2, y2 in normalized_lines:
                x1 = float(x1)
                y1 = float(y1)
                x2 = float(x2)
                y2 = float(y2)

                length = float(np.hypot(x2 - x1, y2 - y1))

                if length < 18:
                    continue

                angle = float(
                    np.degrees(
                        np.arctan2(y2 - y1, x2 - x1)
                    )
                    % 180.0
                )

                line_angles.extend(
                    [angle] * max(1, int(length // 20))
                )

        perimeter = cv2.arcLength(contour, True)
        simplified = cv2.approxPolyDP(
            contour,
            0.018 * perimeter,
            True,
        )

        hull_area = float(
            cv2.contourArea(cv2.convexHull(contour))
        )
        area = float(cv2.contourArea(contour))

        solidity = area / hull_area if hull_area else 0.0

        if len(simplified) > 12 or solidity < 0.72:
            return (
                "complex",
                min(0.76, 0.48 + len(simplified) / 80.0),
                self._dominant_angle(line_angles),
            )

        if len(line_angles) < 4:
            return "flat", 0.48, None

        histogram = Counter(
            int(round(angle / 15.0)) % 12
            for angle in line_angles
        )

        dominant_bins = [
            item
            for item in histogram.most_common(3)
            if item[1] >= 2
        ]

        dominant_angle = self._dominant_angle(line_angles)

        if len(dominant_bins) >= 2:
            first = dominant_bins[0][0] * 15.0
            second = dominant_bins[1][0] * 15.0

            separation = abs(first - second)
            separation = min(
                separation,
                180.0 - separation,
            )

            if 25.0 <= separation <= 90.0:
                return "hipped", 0.62, dominant_angle

        return "pitched", 0.64, dominant_angle

    @staticmethod
    def _dominant_angle(angles: list[float]) -> float | None:
        if not angles:
            return None
        doubled = np.radians(np.asarray(angles) * 2.0)
        mean = np.arctan2(np.mean(np.sin(doubled)), np.mean(np.cos(doubled)))
        return float((np.degrees(mean) / 2.0) % 180.0)

    @staticmethod
    def _contour_orientation(contour: np.ndarray) -> float | None:
        rect = cv2.minAreaRect(contour)
        width, height = rect[1]
        if width <= 0 or height <= 0:
            return None
        angle = float(rect[2])
        if width < height:
            angle += 90.0
        return angle % 180.0

    @staticmethod
    def _orientation_name(angle: float | None) -> str:
        if angle is None:
            return "unknown"
        names = ["E-W", "NE-SW", "N-S", "NW-SE"]
        index = int(((angle + 22.5) % 180.0) // 45.0)
        return names[index]

    @staticmethod
    def _orientation_confidence(contour: np.ndarray, ridge_found: bool) -> float:
        rect = cv2.minAreaRect(contour)
        width, height = rect[1]
        longest = max(width, height)
        shortest = min(width, height)
        elongation = 1.0 - shortest / longest if longest else 0.0
        base = 0.62 if ridge_found else 0.40
        return min(0.84, base + 0.22 * elongation)

    @staticmethod
    def _detect_solar_panels(
        image: np.ndarray, roof_mask: np.ndarray
    ) -> tuple[int, float]:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        dark_blue = cv2.inRange(
            hsv,
            np.array([80, 25, 20], dtype=np.uint8),
            np.array([135, 255, 150], dtype=np.uint8),
        )
        candidate_mask = cv2.bitwise_and(dark_blue, roof_mask)
        contours, _ = cv2.findContours(
            candidate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        count = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if not 18 <= area <= 3000:
                continue
            x, y, width, height = cv2.boundingRect(contour)
            rectangularity = area / float(width * height) if width * height else 0.0
            if rectangularity >= 0.48 and max(width, height) / max(1, min(width, height)) <= 6:
                count += 1
        confidence = 0.70 if count else 0.46
        return count, confidence

    @staticmethod
    def _detect_superstructures(
        image: np.ndarray, roof_mask: np.ndarray
    ) -> tuple[int, float]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        roof_values = gray[roof_mask > 0]
        if roof_values.size == 0:
            return 0, 0.0
        median = float(np.median(roof_values))
        difference = cv2.absdiff(gray, np.full_like(gray, int(median)))
        anomalies = cv2.threshold(difference, 48, 255, cv2.THRESH_BINARY)[1]
        anomalies = cv2.bitwise_and(anomalies, roof_mask)
        anomalies = cv2.morphologyEx(
            anomalies,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        contours, _ = cv2.findContours(
            anomalies, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        roof_area = np.count_nonzero(roof_mask)
        count = sum(
            1
            for contour in contours
            if 15 <= cv2.contourArea(contour) <= max(100.0, roof_area * 0.04)
        )
        return count, 0.43

    @staticmethod
    def _visible_condition(
        image: np.ndarray, roof_mask: np.ndarray
    ) -> tuple[str, float]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        values = gray[roof_mask > 0]
        if values.size < 30:
            return "unknown", 0.0
        coefficient = float(np.std(values) / max(1.0, np.mean(values)))
        if coefficient < 0.16:
            return "visually_uniform", 0.52
        if coefficient > 0.42:
            return "possible_surface_variation", 0.36
        return "no_obvious_issue_visible", 0.40

    @staticmethod
    def _polygon_confidence(contour: np.ndarray, detection_score: float) -> float:
        area = float(cv2.contourArea(contour))
        hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
        solidity = area / hull_area if hull_area else 0.0
        return float(np.clip(0.35 + 0.38 * detection_score + 0.22 * solidity, 0.0, 0.92))

    @staticmethod
    def _build_notes(
        georeference_quality: str,
        roof_type: str,
        material: str,
    ) -> str:
        notes = [
            "Polygon and planimetric area were derived from the GeoTIFF georeferencing.",
            "Slope is null because a single RGB orthophoto does not support a reliable slope measurement without elevation data.",
            "Type, material, solar panels, superstructures and visible condition are image-based heuristic estimates and must be reviewed when confidence is low.",
        ]
        if georeference_quality.lower() == "approximate":
            notes.append(
                "The sample image has approximate georeferencing, so its metric area has reduced confidence."
            )
        if roof_type == "unknown" or material == "unknown":
            notes.append("At least one visual attribute could not be classified reliably.")
        return " ".join(notes)
