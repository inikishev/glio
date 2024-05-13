def seg_json(
    ContentCreatorName="glio_diff (Nikishev I.O. 224-321)",
    ClinicalTrialSeriesID="1",
    ClinicalTrialTimePointID="1",
    SeriesNumber="300",
    InstanceNumber="1",
):
    jsonfile = f"""
{{
  "ContentCreatorName": "{ContentCreatorName}",
  "ClinicalTrialSeriesID": "{ClinicalTrialSeriesID}",
  "ClinicalTrialTimePointID": "{ClinicalTrialTimePointID}",
  "SeriesDescription": "Segmentation",
  "SeriesNumber": "{SeriesNumber}",
  "InstanceNumber": "{InstanceNumber}",
  "BodyPartExamined": "brain",
  "segmentAttributes": [
    [
      {{
        "labelID": 1,
        "SegmentDescription": "edema",
        "SegmentAlgorithmType": "AUTOMATIC",
        "SegmentAlgorithmName": "glio_diff - Никишев Иван Олегович",
        "SegmentedPropertyCategoryCodeSequence": {{
          "CodeValue": "85756007",
          "CodingSchemeDesignator": "SCT",
          "CodeMeaning": "Tissue"
        }},
        "SegmentedPropertyTypeCodeSequence": {{
          "CodeValue": "85756007",
          "CodingSchemeDesignator": "SCT",
          "CodeMeaning": "Tissue"
        }},
        "recommendedDisplayRGBValue": [
          128,
          174,
          128
        ]
      }},
      {{
        "labelID": 2,
        "SegmentDescription": "necrotic core",
        "SegmentAlgorithmType": "AUTOMATIC",
        "SegmentAlgorithmName": "glio_diff - Никишев Иван Олегович",
        "SegmentedPropertyCategoryCodeSequence": {{
          "CodeValue": "85756007",
          "CodingSchemeDesignator": "SCT",
          "CodeMeaning": "Tissue"
        }},
        "SegmentedPropertyTypeCodeSequence": {{
          "CodeValue": "85756007",
          "CodingSchemeDesignator": "SCT",
          "CodeMeaning": "Tissue"
        }},
        "recommendedDisplayRGBValue": [
          128,
          174,
          128
        ]
      }},
      {{
        "labelID": 3,
        "SegmentDescription": "Enhancing tumor",
        "SegmentAlgorithmType": "AUTOMATIC",
        "SegmentAlgorithmName": "glio_diff - Никишев Иван Олегович",
        "SegmentedPropertyCategoryCodeSequence": {{
          "CodeValue": "85756007",
          "CodingSchemeDesignator": "SCT",
          "CodeMeaning": "Tissue"
        }},
        "SegmentedPropertyTypeCodeSequence": {{
          "CodeValue": "85756007",
          "CodingSchemeDesignator": "SCT",
          "CodeMeaning": "Tissue"
        }},
        "recommendedDisplayRGBValue": [
          128,
          174,
          128
        ]
      }}
    ]
  ],
  "ContentLabel": "SEGMENTATION",
  "ContentDescription": "Image segmentation",
  "ClinicalTrialCoordinatingCenterName": "dcmqi"
}}"""

    with open("GLIOBLASTOMA_TEMP.json", 'w', encoding='utf8') as f: f.write(jsonfile)
    return 'GLIOBLASTOMA_TEMP.json'