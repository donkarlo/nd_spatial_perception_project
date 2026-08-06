from __future__ import annotations

from collections import Counter
from math import log1p

import cv2
import numpy as np
from shapely.geometry import Polygon

from .models import BuildingCandidate, GeoImage, RoofResult


class RoofAttributeExtractor:
    """Extract conservative roof attributes from pixels inside one footprint."""

    def extract(
        self,
        geo_image: GeoImage,
        candidate: BuildingCandidate,
        index: int,
        footprint_source_description: str,
    ) -> RoofResult:
        polygon_px = [
            (float(x), float(y))
            for x, y in candidate.polygon_px.exterior.coords
        ]
        polygon_crs = [
            (float(x), float(y))
            for x, y in candidate.polygon_crs.exterior.coords
        ]
        polygon_lonlat = self._crs_points_to_lonlat(
            geo_image=geo_image,
            points=polygon_crs,
        )
        area_m2 = geo_image.georeferencer.area_m2(candidate.polygon_crs)

        (
            panel_count,
            panel_evidence,
            panel_coverage,
            panel_mask,
        ) = self._detect_solar_panels(
            image=geo_image.bgr,
            mask=candidate.mask,
        )

        roof_pixel_count = max(float(np.count_nonzero(candidate.mask)), 1.0)
        residual_mask = candidate.mask.copy()
        if panel_count > 0:
            residual_mask = cv2.bitwise_and(
                candidate.mask,
                cv2.bitwise_not(panel_mask),
            )
        residual_fraction = (
            float(np.count_nonzero(residual_mask)) / roof_pixel_count
        )
        panel_dominated = panel_count > 0 and (
            panel_coverage >= 0.45 or residual_fraction < 0.30
        )

        if panel_dominated:
            material = "unknown"
            material_evidence = 0.0
            roof_type = "unknown"
            type_evidence = 0.0
            ridge_angle = None
            ridge_orientation_evidence = 0.0
        else:
            analysis_mask = (
                residual_mask
                if panel_count > 0 and residual_fraction >= 0.30
                else candidate.mask
            )
            material, material_evidence = self._classify_material(
                image=geo_image.bgr,
                mask=analysis_mask,
            )
            (
                roof_type,
                type_evidence,
                ridge_angle,
                ridge_orientation_evidence,
            ) = self._classify_type(
                image=geo_image.bgr,
                mask=analysis_mask,
                polygon_px=candidate.polygon_px,
            )

        polygon_orientation, polygon_orientation_evidence = (
            self._polygon_orientation(candidate.polygon_px)
        )
        orientation_deg = (
            ridge_angle if ridge_angle is not None else polygon_orientation
        )
        orientation = self._orientation_name(orientation_deg)
        orientation_evidence = (
            ridge_orientation_evidence
            if ridge_angle is not None
            else polygon_orientation_evidence
        )

        if panel_dominated:
            superstructure_count = 0
            superstructure_evidence = 0.0
            visible_condition = "not_assessed_due_to_solar_coverage"
            condition_evidence = 0.0
        else:
            object_mask = (
                residual_mask
                if panel_count > 0 and residual_fraction >= 0.30
                else candidate.mask
            )
            superstructure_count, superstructure_evidence = (
                self._detect_superstructures(
                    image=geo_image.bgr,
                    mask=object_mask,
                )
            )
            visible_condition, condition_evidence = self._visible_condition(
                image=geo_image.bgr,
                mask=object_mask,
            )

        area_confidence = self._area_confidence(candidate)
        material_confidence = self._visual_confidence(
            0.86 * material_evidence,
            candidate.image_quality,
            maximum=0.80,
        )
        type_confidence = self._visual_confidence(
            type_evidence,
            candidate.image_quality,
            maximum=0.84,
        )
        orientation_confidence = self._orientation_confidence(
            orientation_evidence=orientation_evidence,
            boundary_support=candidate.boundary_support,
            image_quality=candidate.image_quality,
            ridge_based=ridge_angle is not None,
        )
        panel_confidence = self._visual_confidence(
            panel_evidence,
            candidate.image_quality,
            maximum=0.86,
        )
        superstructure_confidence = self._visual_confidence(
            superstructure_evidence,
            candidate.image_quality,
            maximum=0.78,
        )
        condition_confidence = self._visual_confidence(
            condition_evidence,
            candidate.image_quality,
            maximum=0.68,
        )

        note_parts = [
            "The polygon is the aligned source footprint and is not assigned a "
            "separate exported confidence score.",
            "Area confidence combines footprint-to-image boundary support, "
            "image quality and raster scale.",
            "Solar-panel count is binary at this orthophoto resolution: 0 means "
            "no accepted array and 1 means solar-panel presence was accepted. "
            "Individual modules or multiple separate arrays are not counted "
            "reliably.",
            "Exact metric roof slope is not recoverable from one RGB orthophoto, "
            "so slope_deg is null and slope confidence is 0.0.",
        ]
        if panel_dominated:
            note_parts.append(
                "A dominant repetitive solar-panel grid obscures most of the roof. "
                "Roof type, underlying material, superstructures and visible roof "
                "condition are therefore reported as unknown or not assessed "
                "instead of being inferred from panel-grid edges."
            )
        else:
            note_parts.append(
                "Visual confidence values are computed from colour, texture, line, "
                "shape and contrast evidence inside the usable roof area."
            )
        notes = " ".join(note_parts)

        return RoofResult(
            building_id=f"building_{index:03d}",
            source_building_id=candidate.source_id,
            source_used=[geo_image.description, footprint_source_description],
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
            visible_condition=visible_condition,
            confidence={
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
    def _crs_points_to_lonlat(
        geo_image: GeoImage,
        points: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        pixel_points = geo_image.georeferencer.crs_to_pixel(points)
        return geo_image.georeferencer.pixel_to_lonlat(pixel_points)

    @staticmethod
    def _masked_pixels(
        image: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return hsv[mask > 0], gray[mask > 0], image[mask > 0]

    @staticmethod
    def _ramp(value: float, low: float, high: float) -> float:
        if high <= low:
            return 0.0
        return float(np.clip((value - low) / (high - low), 0.0, 1.0))

    @classmethod
    def _band_score(
        cls,
        value: float,
        low: float,
        high: float,
        softness: float,
    ) -> float:
        if low <= value <= high:
            return 1.0
        if value < low:
            return cls._ramp(value, low - softness, low)
        return 1.0 - cls._ramp(value, high, high + softness)

    @staticmethod
    def _visual_confidence(
        evidence: float,
        image_quality: float,
        maximum: float,
    ) -> float:
        quality_factor = 0.68 + 0.32 * float(np.clip(image_quality, 0.0, 1.0))
        return float(np.clip(evidence * quality_factor, 0.0, maximum))

    @classmethod
    def _area_confidence(cls, candidate: BuildingCandidate) -> float:
        boundary = float(np.clip(candidate.boundary_support, 0.0, 1.0))
        quality = float(np.clip(candidate.image_quality, 0.0, 1.0))
        pixel_area = max(float(candidate.polygon_px.area), 0.0)
        scale_support = cls._ramp(
            log1p(pixel_area),
            log1p(300.0),
            log1p(10000.0),
        )
        return float(
            np.clip(
                0.46
                + 0.18 * boundary
                + 0.16 * quality
                + 0.12 * scale_support,
                0.0,
                0.93,
            )
        )

    def _classify_material(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[str, float]:
        hsv_pixels, gray_pixels, _ = self._masked_pixels(image, mask)
        if hsv_pixels.size == 0:
            return "unknown", 0.0

        hue = float(np.median(hsv_pixels[:, 0]))
        saturation = float(np.median(hsv_pixels[:, 1]))
        value = float(np.median(hsv_pixels[:, 2]))
        texture = float(np.std(gray_pixels))

        red_fraction = float(
            np.mean(
                ((hsv_pixels[:, 0] <= 28) | (hsv_pixels[:, 0] >= 168))
                & (hsv_pixels[:, 1] >= 45)
            )
        )
        green_fraction = float(
            np.mean(
                (hsv_pixels[:, 0] >= 34)
                & (hsv_pixels[:, 0] <= 92)
                & (hsv_pixels[:, 1] >= 42)
            )
        )

        tiled_score = max(
            0.85 * self._ramp(red_fraction, 0.18, 0.65),
            0.55
            * self._band_score(hue, 0.0, 30.0, 18.0)
            * self._ramp(saturation, 30.0, 100.0),
        )
        tiled_score *= 0.72 + 0.28 * self._ramp(texture, 8.0, 45.0)

        green_score = self._ramp(green_fraction, 0.25, 0.72)
        green_score *= 0.70 + 0.30 * (1.0 - self._ramp(texture, 35.0, 65.0))

        metal_score = (
            self._ramp(value, 145.0, 225.0)
            * (1.0 - self._ramp(saturation, 30.0, 80.0))
            * (1.0 - 0.45 * self._ramp(texture, 20.0, 55.0))
        )
        gravel_score = (
            self._band_score(value, 65.0, 195.0, 55.0)
            * (1.0 - self._ramp(saturation, 35.0, 90.0))
            * (0.65 + 0.35 * self._band_score(texture, 12.0, 48.0, 25.0))
        )
        glass_score = (
            self._ramp(value, 175.0, 245.0)
            * (1.0 - self._ramp(saturation, 30.0, 75.0))
            * self._ramp(texture, 28.0, 70.0)
        )

        scores = {
            "tiled": tiled_score,
            "green_roof": green_score,
            "metal": metal_score,
            "flat/gravel": gravel_score,
            "glass": glass_score,
        }
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        label, best = ranked[0]
        second = ranked[1][1]
        margin = max(0.0, best - second)

        if best < 0.38:
            uncertainty = 1.0 - best
            return "unknown", float(np.clip(0.16 + 0.18 * uncertainty, 0.0, 0.34))

        evidence = 0.34 + 0.38 * best + 0.24 * margin
        return label, float(np.clip(evidence, 0.0, 0.88))

    def _classify_type(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        polygon_px: Polygon,
    ) -> tuple[str, float, float | None, float]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 55, 150)
        eroded_mask = cv2.erode(
            mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        )
        masked_edges = cv2.bitwise_and(edges, edges, mask=eroded_mask)
        roof_pixels = max(float(np.count_nonzero(eroded_mask)), 1.0)
        edge_density = float(np.count_nonzero(masked_edges)) / roof_pixels
        lines = cv2.HoughLinesP(
            masked_edges,
            1,
            np.pi / 180.0,
            threshold=24,
            minLineLength=20,
            maxLineGap=8,
        )

        angles: list[float] = []
        lengths: list[float] = []
        if lines is not None:
            for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
                length = float(np.hypot(x2 - x1, y2 - y1))
                if length < 20.0:
                    continue
                angle = float(
                    np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180.0
                )
                angles.append(angle)
                lengths.append(length)

        if not angles:
            flat_evidence = 0.30 + 0.28 * (
                1.0 - self._ramp(edge_density, 0.025, 0.13)
            )
            return "flat_or_low_pitch", flat_evidence, None, 0.0

        dominant = self._weighted_dominant_angle(angles, lengths)
        histogram: Counter[int] = Counter()
        for angle, length in zip(angles, lengths, strict=True):
            histogram[int(round(angle / 15.0)) % 12] += max(1, int(round(length)))

        total_weight = float(sum(histogram.values()))
        ranked_bins = histogram.most_common(4)
        first_weight = float(ranked_bins[0][1])
        second_weight = float(ranked_bins[1][1]) if len(ranked_bins) > 1 else 0.0
        dominant_share = first_weight / max(total_weight, 1.0)
        second_share = second_weight / max(total_weight, 1.0)
        strong_bins = [item for item in ranked_bins if item[1] / total_weight >= 0.14]

        perimeter = max(float(polygon_px.length), 1.0)
        line_support = self._ramp(sum(lengths) / perimeter, 0.15, 1.8)
        angular_coherence = self._weighted_angular_coherence(angles, lengths)
        orientation_evidence = float(
            np.clip(
                0.28
                + 0.32 * angular_coherence
                + 0.22 * line_support
                + 0.12 * dominant_share,
                0.0,
                0.88,
            )
        )

        if len(strong_bins) >= 3:
            third_share = float(strong_bins[2][1]) / total_weight
            evidence = 0.32 + 0.24 * line_support + 0.22 * min(
                1.0,
                (second_share + third_share) / 0.45,
            ) + 0.12 * self._ramp(edge_density, 0.04, 0.16)
            return "complex", float(np.clip(evidence, 0.0, 0.82)), dominant, orientation_evidence

        if len(strong_bins) >= 2:
            first_angle = strong_bins[0][0] * 15.0
            second_angle = strong_bins[1][0] * 15.0
            separation = min(
                abs(first_angle - second_angle),
                180.0 - abs(first_angle - second_angle),
            )
            separation_support = self._band_score(separation, 30.0, 85.0, 25.0)
            if separation_support >= 0.45:
                balance = 1.0 - min(
                    1.0,
                    abs(first_weight - second_weight) / max(first_weight, 1.0),
                )
                evidence = (
                    0.34
                    + 0.22 * line_support
                    + 0.18 * separation_support
                    + 0.16 * balance
                )
                return "hipped", float(np.clip(evidence, 0.0, 0.84)), dominant, orientation_evidence

        evidence = (
            0.34
            + 0.25 * line_support
            + 0.22 * dominant_share
            + 0.10 * max(0.0, dominant_share - second_share)
        )
        return "pitched", float(np.clip(evidence, 0.0, 0.82)), dominant, orientation_evidence

    @staticmethod
    def _weighted_dominant_angle(
        angles: list[float],
        lengths: list[float],
    ) -> float | None:
        if not angles:
            return None
        doubled = np.radians(np.asarray(angles, dtype=np.float64) * 2.0)
        weights = np.asarray(lengths, dtype=np.float64)
        mean = np.arctan2(
            float(np.average(np.sin(doubled), weights=weights)),
            float(np.average(np.cos(doubled), weights=weights)),
        )
        return float((np.degrees(mean) / 2.0) % 180.0)

    @staticmethod
    def _weighted_angular_coherence(
        angles: list[float],
        lengths: list[float],
    ) -> float:
        doubled = np.radians(np.asarray(angles, dtype=np.float64) * 2.0)
        weights = np.asarray(lengths, dtype=np.float64)
        x = float(np.average(np.cos(doubled), weights=weights))
        y = float(np.average(np.sin(doubled), weights=weights))
        return float(np.clip(np.hypot(x, y), 0.0, 1.0))

    @staticmethod
    def _polygon_orientation(polygon: Polygon) -> tuple[float | None, float]:
        contour = np.round(
            np.asarray(polygon.exterior.coords)
        ).astype(np.float32).reshape(-1, 1, 2)
        if len(contour) < 3:
            return None, 0.0
        rectangle = cv2.minAreaRect(contour)
        width, height = rectangle[1]
        if width <= 0.0 or height <= 0.0:
            return None, 0.0
        angle = float(rectangle[2])
        if width < height:
            angle += 90.0
        elongation = max(width, height) / max(min(width, height), 1e-6)
        elongation_evidence = float(np.clip((elongation - 1.0) / 2.5, 0.0, 1.0))
        rectangularity = float(
            np.clip(float(polygon.area) / max(width * height, 1.0), 0.0, 1.0)
        )
        evidence = 0.22 + 0.38 * elongation_evidence + 0.18 * rectangularity
        return angle % 180.0, float(np.clip(evidence, 0.0, 0.72))

    @staticmethod
    def _orientation_confidence(
        orientation_evidence: float,
        boundary_support: float,
        image_quality: float,
        ridge_based: bool,
    ) -> float:
        source_bonus = 0.08 if ridge_based else 0.0
        confidence = (
            0.58 * orientation_evidence
            + 0.18 * float(np.clip(boundary_support, 0.0, 1.0))
            + 0.12 * float(np.clip(image_quality, 0.0, 1.0))
            + source_bonus
        )
        return float(np.clip(confidence, 0.0, 0.86))

    @staticmethod
    def _orientation_name(angle: float | None) -> str:
        if angle is None:
            return "unknown"
        names = ["E-W", "NE-SW", "N-S", "NW-SE"]
        index = int(((angle + 22.5) % 180.0) // 45.0)
        return names[index]

    def _detect_solar_panels(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[int, float, float, np.ndarray]:
        """Detect visible solar-panel arrays conservatively.

        At this orthophoto resolution, individual modules and multiple separate
        arrays cannot be counted reliably. The returned count is therefore binary:
        0 for no accepted solar array and 1 for accepted solar-panel presence.
        Detection requires a repeated rectilinear grid plus dark, low-saturation
        surface evidence; blue-grey colour alone is not enough because metal roofs
        can have a similar colour.
        """

        roof_area = max(float(np.count_nonzero(mask)), 1.0)
        empty_mask = np.zeros_like(mask)
        if roof_area < 120.0:
            return 0, 0.0, 0.0, empty_mask

        grid_evidence = self._solar_grid_evidence(image=image, mask=mask)
        detection_threshold = 0.72
        if grid_evidence < detection_threshold:
            # Confidence is in the reported absence, not a probability of
            # presence. Near-threshold negatives intentionally remain uncertain.
            absence_evidence = 0.22 + 0.42 * (1.0 - grid_evidence)
            return (
                0,
                float(np.clip(absence_evidence, 0.0, 0.64)),
                0.0,
                empty_mask,
            )

        panel_mask = self._solar_panel_mask(image=image, mask=mask)
        coverage = float(np.count_nonzero(panel_mask)) / roof_area
        if coverage < 0.15:
            # The grid signal was strong, but there is not enough coherent dark
            # surface support to claim a panel array.
            uncertain_absence = 0.18 + 0.24 * (1.0 - grid_evidence)
            return 0, uncertain_absence, 0.0, empty_mask

        positive_evidence = 0.46 + 0.45 * grid_evidence
        return (
            1,
            float(np.clip(positive_evidence, 0.0, 0.88)),
            float(np.clip(coverage, 0.0, 1.0)),
            panel_mask,
        )

    def _solar_grid_evidence(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> float:
        patch, patch_mask = self._rectified_roof_patch(
            image=image,
            mask=mask,
        )
        if patch.size == 0 or patch_mask.size == 0:
            return 0.0

        valid = patch_mask > 0
        if np.count_nonzero(valid) < 120:
            return 0.0

        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        hsv_pixels = hsv[valid]

        median_saturation = float(np.median(hsv_pixels[:, 1]))
        dark_neutral_fraction = float(
            np.mean(
                (hsv_pixels[:, 1] <= 120)
                & (hsv_pixels[:, 2] >= 25)
                & (hsv_pixels[:, 2] <= 190)
            )
        )

        eroded_mask = cv2.erode(
            patch_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        eroded_pixels = max(float(np.count_nonzero(eroded_mask)), 1.0)
        edges = cv2.Canny(gray, 40, 120)
        masked_edges = cv2.bitwise_and(edges, edges, mask=eroded_mask)
        edge_density = float(np.count_nonzero(masked_edges)) / eroded_pixels

        gradient_x = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
        gradient_y = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
        valid_eroded = eroded_mask > 0
        column_profile = np.sum(
            gradient_x * valid_eroded,
            axis=0,
        ) / np.maximum(np.sum(valid_eroded, axis=0), 1)
        row_profile = np.sum(
            gradient_y * valid_eroded,
            axis=1,
        ) / np.maximum(np.sum(valid_eroded, axis=1), 1)

        column_periodicity = self._projection_periodicity(column_profile)
        row_periodicity = self._projection_periodicity(row_profile)
        strongest_periodicity = max(column_periodicity, row_periodicity)

        lines = cv2.HoughLinesP(
            masked_edges,
            1,
            np.pi / 180.0,
            threshold=12,
            minLineLength=8,
            maxLineGap=4,
        )
        line_count = 0 if lines is None else int(len(lines))

        evidence = (
            0.42 * self._ramp(strongest_periodicity, 0.22, 0.52)
            + 0.18 * (1.0 - self._ramp(median_saturation, 18.0, 70.0))
            + 0.18 * self._ramp(edge_density, 0.16, 0.34)
            + 0.12 * self._ramp(dark_neutral_fraction, 0.45, 0.90)
            + 0.10 * self._ramp(float(line_count), 20.0, 55.0)
        )
        return float(np.clip(evidence, 0.0, 1.0))

    @staticmethod
    def _projection_periodicity(profile: np.ndarray) -> float:
        values = np.asarray(profile, dtype=np.float64)
        if values.size < 12 or float(np.std(values)) < 1e-6:
            return 0.0

        values = values - float(np.mean(values))
        autocorrelation = np.correlate(values, values, mode="full")[
            values.size - 1 :
        ]
        if autocorrelation.size == 0 or autocorrelation[0] <= 0.0:
            return 0.0
        autocorrelation = autocorrelation / autocorrelation[0]

        first_lag = 3
        final_lag = min(values.size // 2, 25)
        if final_lag <= first_lag:
            return 0.0
        return float(
            np.clip(
                np.max(autocorrelation[first_lag:final_lag]),
                0.0,
                1.0,
            )
        )

    @staticmethod
    def _rectified_roof_patch(
        image: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            return np.empty((0, 0, 3), dtype=image.dtype), np.empty(
                (0, 0), dtype=np.uint8
            )

        contour = max(contours, key=cv2.contourArea)
        x, y, width, height = cv2.boundingRect(contour)
        padding = 10
        x0 = max(0, x - padding)
        y0 = max(0, y - padding)
        x1 = min(image.shape[1], x + width + padding)
        y1 = min(image.shape[0], y + height + padding)
        patch = image[y0:y1, x0:x1].copy()
        patch_mask = mask[y0:y1, x0:x1].copy()

        local_contours, _ = cv2.findContours(
            patch_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if not local_contours:
            return patch, patch_mask
        local_contour = max(local_contours, key=cv2.contourArea)
        rectangle = cv2.minAreaRect(local_contour)
        center = rectangle[0]
        rectangle_width, rectangle_height = rectangle[1]
        angle = float(rectangle[2])
        if rectangle_width < rectangle_height:
            angle += 90.0

        transform = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(
            patch,
            transform,
            (patch.shape[1], patch.shape[0]),
            flags=cv2.INTER_LINEAR,
        )
        rotated_mask = cv2.warpAffine(
            patch_mask,
            transform,
            (patch.shape[1], patch.shape[0]),
            flags=cv2.INTER_NEAREST,
        )
        nonzero = cv2.findNonZero(rotated_mask)
        if nonzero is None:
            return patch, patch_mask
        rx, ry, rw, rh = cv2.boundingRect(nonzero)
        return (
            rotated_image[ry : ry + rh, rx : rx + rw],
            rotated_mask[ry : ry + rh, rx : rx + rw],
        )

    @staticmethod
    def _solar_panel_mask(
        image: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0]
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]

        neutral_dark = (
            (saturation <= 45)
            & (value >= 45)
            & (value <= 195)
        )
        blue_dark = (
            (hue >= 85)
            & (hue <= 145)
            & (saturation >= 15)
            & (value >= 35)
            & (value <= 185)
        )
        candidate = np.where(
            (neutral_dark | blue_dark) & (mask > 0),
            255,
            0,
        ).astype(np.uint8)
        candidate = cv2.morphologyEx(
            candidate,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        )
        candidate = cv2.morphologyEx(
            candidate,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        )
        candidate = cv2.bitwise_and(candidate, mask)

        component_count, labels, statistics, _ = (
            cv2.connectedComponentsWithStats(candidate, connectivity=8)
        )
        roof_area = max(float(np.count_nonzero(mask)), 1.0)
        result = np.zeros_like(mask)
        minimum_component_area = max(20.0, 0.03 * roof_area)
        for component_index in range(1, component_count):
            component_area = float(
                statistics[component_index, cv2.CC_STAT_AREA]
            )
            if component_area >= minimum_component_area:
                result[labels == component_index] = 255
        return result

    def _detect_superstructures(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[int, float]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        local_mean = cv2.blur(gray, (17, 17))
        deviation = cv2.absdiff(gray, local_mean)
        high_contrast = cv2.threshold(
            deviation,
            34,
            255,
            cv2.THRESH_BINARY,
        )[1]
        high_contrast = cv2.bitwise_and(high_contrast, mask)
        contours, _ = cv2.findContours(
            high_contrast,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        roof_area = max(float(np.count_nonzero(mask)), 1.0)
        maximum_object_area = max(250.0, roof_area * 0.08)
        accepted_areas = [
            float(cv2.contourArea(contour))
            for contour in contours
            if 18.0 <= float(cv2.contourArea(contour)) <= maximum_object_area
        ]
        count = min(len(accepted_areas), 20)
        contrast_fraction = float(np.count_nonzero(high_contrast)) / roof_area

        if count > 0:
            object_area_share = sum(accepted_areas) / roof_area
            evidence = (
                0.30
                + 0.20 * self._ramp(float(count), 1.0, 8.0)
                + 0.20 * self._ramp(object_area_share, 0.003, 0.08)
                + 0.14 * self._band_score(contrast_fraction, 0.01, 0.16, 0.12)
            )
        else:
            clean_absence = 1.0 - self._ramp(contrast_fraction, 0.01, 0.16)
            evidence = 0.26 + 0.24 * clean_absence
        return count, float(np.clip(evidence, 0.0, 0.78))

    def _visible_condition(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[str, float]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        pixels = gray[mask > 0]
        if pixels.size == 0:
            return "not_assessed", 0.0

        dark_fraction = float(np.mean(pixels < 40))
        bright_fraction = float(np.mean(pixels > 235))
        extreme_fraction = dark_fraction + bright_fraction
        texture = float(np.std(pixels))
        contrast_risk = self._ramp(extreme_fraction, 0.08, 0.32)
        texture_risk = self._ramp(texture, 35.0, 75.0)

        if extreme_fraction > 0.20 or texture > 62.0:
            severity = max(contrast_risk, texture_risk)
            evidence = 0.30 + 0.30 * severity
            return "partly_obscured_or_high_contrast", float(
                np.clip(evidence, 0.0, 0.64)
            )

        clean_visibility = 1.0 - max(contrast_risk, 0.65 * texture_risk)
        evidence = 0.28 + 0.30 * clean_visibility
        return "no_obvious_issue_visible", float(np.clip(evidence, 0.0, 0.60))
