from llm.vision.models.vision_result import DataPoint, VisionResult


def test_vision_result():

    result = VisionResult(
        image_type="line_chart",
        description="Annual revenue increased over time.",
        data_points=[
            DataPoint(label="2022", value="₹6 Cr"),
            DataPoint(label="2023", value="₹8 Cr"),
        ],
        entities=["Acme Technologies"],
        keywords=["revenue", "growth"],
    )

    assert result.image_type == "line_chart"
    assert len(result.data_points) == 2
    assert result.data_points[0].label == "2022"
    assert result.data_points[0].value == "₹6 Cr"
    assert "Acme Technologies" in result.entities
    assert "revenue" in result.keywords