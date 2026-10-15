# AI-Powered Medical Named Entity Recognition

A Streamlit application that sends English clinical text to the official AWS
Comprehend Medical API. It displays AWS-returned entities, character offsets,
confidence scores, traits, attributes, relationships, and optional ICD-10-CM
and RxNorm concept candidates.

This is an educational NLP tool, not a clinical decision support system. It does
not diagnose, recommend treatment, or make this application HIPAA-compliant.
Avoid sending identifiable patient information unless you are authorized and
have independently reviewed your AWS account and applicable privacy controls.
Amazon Comprehend Medical usage may incur AWS charges.

## Run in Windows PowerShell / VS Code

Open this `medical_ner` folder in VS Code, then run these commands in its
terminal:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Configure AWS credentials using a named profile (recommended) or the standard
AWS credential chain, then launch the app:

```powershell
aws configure --profile medical-ner
$env:AWS_PROFILE = "medical-ner"
streamlit run app.py
```

Streamlit prints the local browser URL, normally `http://localhost:8501`. Stop
the app with `Ctrl+C`. If PowerShell blocks virtual environment activation,
use `.\.venv\Scripts\python.exe -m pip install -r requirements.txt` and
`.\.venv\Scripts\python.exe -m streamlit run app.py`.

The app loads non-secret settings from `.env`. Set the service region:

```dotenv
AWS_REGION=us-east-1
AWS_PROFILE=medical-ner
```

Do not put AWS access keys in `.env`, source code, Streamlit inputs, or reports.
Use `aws configure`, environment credentials managed outside the app, or an IAM
role. `.env` is ignored by Git. `.env.example` contains no credentials.

## AWS setup, access, and costs

1. Use an AWS account where Amazon Comprehend Medical is available.
2. Choose a supported AWS region and set `AWS_REGION`. Check the current
   [regional services list](https://aws.amazon.com/about-aws/global-infrastructure/regional-product-services/)
   if the selected region is rejected. Availability and enabled operations can
   differ by region.
3. Configure a least-privilege IAM identity. Grant only the actions used:
   `comprehendmedical:DetectEntitiesV2`,
   `comprehendmedical:InferICD10CM`, and
   `comprehendmedical:InferRxNorm`. Use the
   [AWS Service Authorization Reference](https://docs.aws.amazon.com/service-authorization/latest/reference/list_amazoncomprehendmedical.html)
   to confirm the current actions and resource support.
4. Configure a local profile with `aws configure --profile medical-ner`, or
   attach an appropriate IAM role in a managed runtime.
5. The Home dashboard checks local credential discovery and whether the
   selected region can construct a client. **Test AWS connection** makes a real
   request with synthetic text and may be billable.
6. Review current AWS pricing before use:
   [Amazon Comprehend Medical pricing](https://aws.amazon.com/comprehend/medical/pricing/).
   Detection and concept lookup calls can incur charges; this software does
   not claim that AWS usage is free.

Official API references:

- [DetectEntitiesV2](https://docs.aws.amazon.com/comprehend-medical/latest/api/API_DetectEntitiesV2.html)
- [InferICD10CM](https://docs.aws.amazon.com/comprehend-medical/latest/api/API_InferICD10CM.html)
- [InferRxNorm](https://docs.aws.amazon.com/comprehend-medical/latest/api/API_InferRxNorm.html)

## Operations and limits

- `DetectEntitiesV2` is the primary extraction operation. AWS documents a
  non-empty English text input smaller than 20,000 UTF-8 bytes; the UI enforces
  that limit before sending.
- `InferICD10CM` and `InferRxNorm` accept up to 10,000 characters. These
  optional operations are separate requests and may add charges.
- AWS entity categories and types are shown separately from the application's
  derived display groups. The app does not invent codes, confidence values, or
  relations.
- Concept matches are candidates only—not confirmed diagnoses, prescriptions,
  or billing determinations.
- Comprehend Medical supports English text; other languages are not translated
  automatically.

## Privacy and safe use

The app uses synthetic notes by default and asks for confirmation before an
analysis request. Text is sent to AWS for processing. The app does not persist
clinical notes or write raw text or credentials to logs. AWS service processing
is governed by your AWS agreement and configuration. Do not treat this demo as
HIPAA-compliant or as a substitute for privacy/security review.

The tool extracts text spans; it does not provide medical advice. A qualified
professional must review results before any clinical, billing, or coding use.

## Project layout

```text
medical_ner/
├── app.py
├── config/
├── core/                 # validation, mapping, highlighting, analytics
├── services/             # boto3 Comprehend Medical calls
├── ui/                   # dashboard, results, visualizations
├── utils/                # export and safe error messages
├── samples/              # synthetic sample text
├── tests/                # unit tests with mocked AWS responses
├── .env.example
└── requirements.txt
```

## Test without AWS calls

```powershell
python -m pytest tests -q
```

The tests mock AWS SDK responses and do not require credentials or make billable
requests.

## Troubleshooting

- **Missing credentials:** run `aws configure --profile medical-ner`, set
  `$env:AWS_PROFILE`, or use the normal AWS role/credential chain.
- **Access denied:** ask an AWS administrator to grant the required
  `comprehendmedical:*` actions listed above.
- **Invalid/unsupported region:** set `AWS_REGION` to a region currently
  offering Comprehend Medical; the app does not silently change regions.
- **Throttling:** wait and retry; SDK retries are enabled with bounded
  backoff.
- **Input too long:** shorten the input; do not split clinical text without
  considering that cross-segment relationships may be lost.
- **Missing package:** activate the virtual environment and run
  `python -m pip install -r requirements.txt`.
- **No entities:** the service may return an empty entity list; no local
  regex-based entity predictions are added as substitutes.
