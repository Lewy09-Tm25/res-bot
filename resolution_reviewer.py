import os
import json
from docx import Document
from openai import OpenAI
from typing import Dict
import tiktoken

def count_tokens(text: str) -> int:
    encoding = tiktoken.encoding_for_model("gpt-4")
    return len(encoding.encode(text))

class ResolutionReviewer:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.system_prompt = self._get_system_prompt()

    def _get_system_prompt(self) -> str:
        return """You are an expert reviewer of CUNY Board of Trustees resolutions. Your task is to analyze resolutions for compliance with the template and rules below.

For each violation found, you must:
1. Identify the specific rule or template requirement that was violated
2. Explain the error clearly
3. Provide the exact line where the error occurs. If a WHEREAS clause doesn't address a specific point that it is supposed to, then provide the number of that clause.
4. Suggest how to fix it

TEMPLATE STRUCTURE:
1. The resolution must follow this exact structure:
   - First line: "Board of Trustees of The City University of New York"
   - Next line: "RESOLUTION TO"
   - Next line: "Establish a [Degree Level Program] in [Subject] at [College Name]"
   - Next line: [Date of Board of Trustees' committee meeting]. Date must be in format "Month DD, YYYY"
   - WHEREAS clauses
   - 'NOW THEREFORE BE IT'
   - RESOLVED clause
   - EXPLANATION part

2. WHEREAS Clauses Requirements:
   - Each WHEREAS clause, must address specific points, individually and in order:
     1. Why the CUNY and market needs this program
     2. How the program curriculum and credits are designed to meet the needs of the CUNY and market
     3. Current student interest in the program
     4. Transferability of the program courses
     5. Benefits to students served and benefits to the CUNY and market
     6. Projected enrollment, retention, and graduation in the first 3 to 5 years
     7. Financial sustainability of the program given demand, section size, and current staffing, and projected revenue through enrollment to cover all operating costs in the first year
     8. Program investments during the initial growth period; include any space/equipment needs, renovations, staff and faculty hiring (list number of Part and Full Time Faculty), as indicated, and identify the sources of funds for each.
   - If a clause doesn't address a specific point that it is supposed to in a clear manner, then it is a violation and you must provide the number of that clause, and the specific point it doesn't address, along with the suggested fix.
   - All the specific points must be addressed in a clear and concise manner. If not, then it is a violation and you must provide the point that is missing, and suggest/write a WHEREAS clause that addresses it.
   - Each clause must end with "; and" except the last one which ends with a period
   - Each clause must be a single statement without any full-stops. Internal periods, like commas, are allowed.

3. RESOLVED Clause Requirements:
   - The RESOLVED clause must state the aim of the resolution succinctly
   - The RESOLVED clause must be a single statement with only 1 full-stop. Internal periods, like commas, are allowed.

4. EXPLANATION Requirements:
   - The EXPLANATION part must briefly summarize the purpose and benefits of the resolution.

5. Resolution Structure Rules:
   - Must include "NOW, THEREFORE, BE IT" after the last WHEREAS clause
   - Must have exactly one RESOLVED clause
   - Must include EXPLANATION section after RESOLVED clause

FORMATTING RULES:
1. Clause Formatting:
   - WHEREAS clauses must start with "WHEREAS,"
   - Only one full-stop allowed across all WHEREAS clause (except the last WHEREAS clause)
   - "; and" required at end of all WHEREAS clauses except the last
   - RESOLVED clause must start with "RESOLVED,"
   - RESOLVED clause must come after "NOW, THEREFORE, BE IT"
   - EXPLANATION part must start with "EXPLANATION:"
   - EXPLANATION must come after RESOLVED clause

Provide your analysis in the following JSON format:
{
    "template_violations": [
        {
            "rule": "string - the specific rule violated",
            "location": "string - where in the document" OR "#clause that violates a rule - the number of the clause violated - the specific point it doesn't address",
            "description": "string - clear explanation of the violation",
            "suggestion": "string - how to fix it"
        }
    ],
    "formatting_violations": [
        {
            "rule": "string - the specific rule violated",
            "location": "string - where in the document",
            "description": "string - clear explanation of the violation",
            "suggestion": "string - how to fix it"
        }
    ],
    "overall_assessment": "string - brief summary of the resolution's compliance"
}
"""

    def read_document(self, file_path: str) -> str:
        doc = Document(file_path)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])

    def review_resolution(self, file_path: str) -> Dict:
        resolution_text = self.read_document(file_path)
        ex_original, ex_modified, changes = self._get_examples()
        user_prompt = self._build_user_prompt(ex_original, ex_modified, changes, resolution_text)
        system_tokens = count_tokens(self.system_prompt)
        user_tokens = count_tokens(user_prompt)
        total_tokens = system_tokens + user_tokens
        response = self.client.chat.completions.create(
            model="gpt-4.1-mini-2025-04-14",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={ "type": "json_object" }
        )
        print("DEBUG: response.choices[0].message.content =", repr(response.choices[0].message.content))
        return json.loads(response.choices[0].message.content)

    def _get_examples(self):
        ex_original = """
        Board of Trustees of the City University of New York

        RESOLUTION TO

        Establish a Bachelor of Arts in Data Analytics at Brooklyn College

        February 28, 2025

        WHEREAS, Knowledge of data practices, ranging from programming to statistics to data story-telling and visualization, is in high demand in the contemporary labor market, according to the US Department of Labor, with expected growth in New York State between 10% and 43% over the next ten years; and

        WHEREAS, The typical approach to undergraduate programs in the field (including data analytics in various forms as well as data science) has tended to stress technical skills. Much less attention has been paid in undergraduate programs to a broad perspective—data practices in society, or the "data landscape"—and the integration of data skills and training in contexts, organizations, and communities; and

        WHEREAS, Brooklyn College is proposing the establishment of a Bachelor of Arts ("BA") program in Data Analytics, organized around the idea of data acumen, or the ability to make creative, sound judgments and decisions with data. This approach requires a solid foundation in data skills, such as programming in Python, statistics and probability, and data visualization. But it goes beyond this foundation by bringing to bear deep knowledge of communication and contexts from the social and behavioral sciences, and;

        WHEREAS, The proposed 58- to 72.5-credit Bachelor of Arts program tracks to capture a diverse population of students across the behavioral, natural, and social sciences. The first track is for social and behavioral science or humanities students who are interested in careers in data analytics. Students in this track take statistics and data analysis courses in the department of Management, Marketing, and Entrepreneurship, Economics, Psychology, or Sociology. The second track is intended for STEM students who are interested in more mathematics-oriented careers in data science; and

        WHEREAS, The proposed Bachelor of Arts program will be housed in the School of Natural and Behavioral Sciences with participation of the Departments of Computer and Informational Science, Mathematics, Economics, Management, Marketing and Entrepreneurship, Psychology, Sociology, and Communication Arts, Sciences and Disorders; and

        WHEREAS, Students completing the program will acquire the knowledge and skills necessary for career advancement in areas of data management, data analytics, data visualization, and data communications in a variety of employment settings; and

        WHEREAS, The proposed Bachelor of Arts program has an articulation agreement with the Associate of Science program in data science at Borough of Manhattan Community College. Further, possible connections with other University programs at the undergraduate and graduate levels, as well as programs of the City of New York to promote data careers, ensures a reliable pipeline of students into the professions; and

        WHEREAS, The predicted enrollment and retention of students in the BA in Data Analytics, based on existing student interest as well as substantial and expanding market demand for graduates with this qualification, is expected to generate robust growth. The program will increase revenue while containing costs through the use of existing faculty, curriculum, and college resources; and

        NOW, THEREFORE, BE IT

        RESOLVED, That the Board of Trustees of the City University of New York
        authorizes the proposed program in Data Analytics leading to the Bachelor of Arts degree at Brooklyn College be presented to the New York State Education Department for their consideration and registration in accordance with any and all regulations of the New York State Department of Education ("NYSED") for their consideration and registration in accordance with any and all of NYSED's regulations, subject to financial ability. 

        EXPLANATION: The proposed program will build upon a strong foundation in data analysis and quantitative methods, drawing on existing faculty expertise across three schools at Brooklyn College. It will serve The City University of New York's mission to prepare its diverse population of students for the future of work in data careers. It will allow students to develop the necessary skills for academic and professional advancement in this fast-growing area while ensuring equity and access to this vital professional field.
        """
        ex_modified = """
        Board of Trustees of the City University of New York

        RESOLUTION TO

        Establish a Bachelor of Arts in Data Analytics at Brooklyn College

        February 28, 2025

        WHEREAS, In the contemporary labor market, knowledge of data practices, everything from programming to statistics to data story-telling and visualization, is in high demand, according to the US Department of Labor, with expected growth in New York State between 10% and 43% over the next ten years; and

        WHEREAS, The typical approach to undergraduate programs in the field (including data analytics in various forms as well as data science) has tended to stress technical skills with less attention focusing on designing undergraduate programs that take a broad perspective—data practices in society, or the "data landscape"—and the integration of data skills and training in contexts, organizations, and communities; and

        WHEREAS, Brooklyn College is proposing the establishment of a Bachelor of Arts ("BA") program in Data Analytics, organized around the idea of data acumen, or the ability to make creative, sound judgments and decisions with data taking an approach requires a solid foundation in data skills, such as programming in Python, statistics and probability, and data visualization but also goes beyond this foundation by bringing to bear deep knowledge of communication and contexts from the social and behavioral sciences, and;

        WHEREAS, The proposed 58- to 72.5-credit Bachelor of Arts program provides students with the capability to acquire knowledge and skills necessary for career development in areas of data management, data analytics, data visualization, and data communications in a variety of employment settings make possible though the inclusion of two tracks capturing a diverse population of students with the first track targeting social and behavioral science or humanities students who are interested in careers in data analytics with them completing courses in statistics and data analysis and the second track targeting STEM students who are interested in more mathematics-oriented careers in data science; and

        WHEREAS, The proposed Bachelor of Arts program is interdisciplinary being housed in the School of Natural and Behavioral Sciences and in partnership with the Departments of Computer and Informational Science, Mathematics, Economics, Management, Marketing and Entrepreneurship, Psychology, Sociology, and Communication Arts, Sciences and Disorders; and

        WHEREAS, The proposed Bachelor of Arts program has an articulation agreement with the Associate of Science program in data science at Borough of Manhattan Community College with possible connections with other University programs at the undergraduate and graduate levels, as well as programs of the City of New York to promote data careers, ensuring a reliable pipeline of students into the professions; and

        WHEREAS, The predicted enrollment and retention of students in the Bachelor of Arts in Data Analytics, based on existing student interest as well as substantial and expanding market demand for graduates with this qualification, is expected to generate robust growth and the program will increase revenue while containing costs through the use of existing faculty, curricula, and college resources.

        NOW, THEREFORE, BE IT

        RESOLVED, That the Board of Trustees of the City University of New York
        authorizes the proposed program in Data Analytics leading to the Bachelor of Arts degree at Brooklyn College be presented to the New York State Education Department for their consideration and registration in accordance with any and all regulations of the New York State Department of Education ("NYSED") for their consideration and registration in accordance with any and all of NYSED's regulations, subject to financial ability. 

        EXPLANATION: The proposed program will build upon a strong foundation in data analysis and quantitative methods, drawing on existing faculty expertise across three schools at Brooklyn College. It will serve The City University of New York's mission to prepare its diverse population of students for the future of work in data careers as well as help fill the growing market demand for professionals with the skills and background in diverse data practices. It will allow students to develop the necessary skills for academic and professional advancement in this fast-growing area while ensuring equity and access to this vital professional field by targeting students across a range of disciplinary backgrounds from humanities to social sciences to STEM.
        """
        changes = """

        In the first 'WHEREAS' clause, the positiong of the phrase 'In the contemporary labor market,' is changed, inserting it first instead of writing it last, removing 'K'. 'everything' is replaced with 'ranging'.

        In the second 'WHEREAS' clause, there is a period (full-stop), violating a rule. So 'Much' has been replaced by 'with'. 'has been paid' has been replaced with 'focusing on designing'. 'to a' has been replaced with 'that take a'.

        In the third 'WHEREAS' clause, there are periods (full-stop), that violates a rule. So 'with data. This' is replaced with 'with data taking an'. 'that' is added for proper sentence structuing. 'data visualization. But it' is replaced with 'data visualization but also'.

        In the fourth 'WHEREAS' clause, there are periods (full-stop), that violates a rule. The paragraph is also formatted properly. The text 'provides students with the capability to acquire knowledge and skills necessary for career development in areas of data management, data analytics, data visualization, and data communications in a variety of employment settings make possible though the inclusion of' is the sixth 'WHEREAS' claues in the original, but is inserted at the beginning of this clause. Multiple grammatical and syntactic changes are also made.

        In the fifth 'WHEREAS' clause, certain formatting is done, both grammatical and syntactic.

        The sixth 'WHEREAS' clause is completely removed, and inserted in the beginning of the fourth 'WHEREAS' clause.

        The seventh 'WHEREAS' clause has a period (full-stop) that is dealt with accordingly, keeping the rest of the content same.

        The eight 'WHEREAS' clause has a period (full-stop) and it ends with '; and' which are violations. Minor grammatical changes are done. Also, the full-form of 'BA' instead of the abbreviation is also done, indicating professionalism of the resolution.

        The 'NOW, THEREFORE, BE IT' is correct.

        The 'RESOLVED,' clause is also correct.

        In the 'EXPLANATION' part, two sentences are added at the end, which are 'as well as help fill the growing market demand for professionals with the skills and background in diverse data practices' and ' by targeting students across a range of disciplinary backgrounds from humanities to social sciences to STEM'.
        
        SUMMARY:- Overall there are multiple modifications needed in the original version, including grammatical and syntactic changes. Certain rules are violated, which are handled accordingly. In general, the original resolution broadly talks in accordance with the template rules and address the points in order.
        """
        return ex_original, ex_modified, changes

    def _build_user_prompt(self, ex_original, ex_modified, changes, resolution_text):
        return f"""
        Your task is to analyze a draft resolution for compliance with the template and rules. 
        An example of a draft resolution is given below, and the suggested corrections. 
        The incorrect resolution is given below, followed by the updated version of that resolution. 
        The changes that were made and the rules violated are given after the updated version.
        <EXAMPLE START>
        ORIGINAL VERSION:-
        {ex_original}
        ----------------
        MODIFIED VERSION:-
        {ex_modified}
        ----------------
        ORIGINAL VERSION:-
        {changes}
        ----------------
        <EXAMPLE END>
        
        Based on the template provided, alonside the rules that may or may not be violated, please analyze the following resolution for compliance with the template and rules:
        {resolution_text}

        Provide a detailed analysis identifying any violations of the template or rules.""" 