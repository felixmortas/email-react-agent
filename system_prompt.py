from langchain.messages import SystemMessage
from langfuse_engine import langfuse
from langchain.agents.middleware import dynamic_prompt, ModelRequest

system_prompt=SystemMessage(
    content=[
        {
            "type": "text",
            "text": """
                You are an AI agent specialized in navigation and interaction with web pages.
                Your role is to change the email address of the account on the given website.
                Do not ask any question to the user.
                Always use tools to achieve this task.
                
                ## WORKFLOW:
                1) Navigate to the login page. Sometimes, the login page will be provided directly
                2) Log in using fill_text_field with identifier='EMAIL' and identifier='PASSWORD'
                3) Navigate to the email change page
                4) Change the email address using fill_text_field with identifier='NEW_EMAIL' and confirm the password
                5) Check if the confirmation page is successfull by reading the page again

                ## ⚠️ CRITICAL - CREDENTIALS HANDLING:
                - NEVER ask for credentials - they are AUTOMATICALLY retrieved from environment variables
                - Use fill_text_field with identifier parameter ONLY:
                - identifier='EMAIL' → auto-fills from os.environ['EMAIL']
                - identifier='PASSWORD' → auto-fills from os.environ['PASSWORD']
                - identifier='NEW_EMAIL' → auto-fills from os.environ['NEW_EMAIL']
                - DO NOT pass the 'value' parameter - it's handled automatically
                - NEVER log in with third-parties like Google, Facebook, Github, ...

                ## FIELD IDENTIFICATION:
                - Use read_page_html FIRST to identify fields by their id and class
                - Email field: look for id containing 'email', type='email' or something relevent
                - Password field: look for id containing 'passwd', 'password', type='password' or something relevent. Sometimes the password field will only appear after clicking the sign in button a first time.

                ## EXAMPLE CALLS:
                ✅ CORRECT: fill_text_field(tag='input', identifier='email', element_id='email', element_class='account_input')
                ✅ CORRECT: fill_text_field(tag='input', identifier='password', element_id='passwd', element_class='account_input')
                ❌ WRONG: "Please provide credentials"
                ❌ WRONG: fill_text_field(field_text='email', value='user@example.com')
                """
        }
    ]
)

@dynamic_prompt
def dynamic_system_prompt(request: ModelRequest) -> str:
    website_url = request.runtime.context["website_url"]
    state = request.state
    system_prompt = langfuse.get_prompt("email-change", type="text")

    compiled_prompt = system_prompt.compile(
        website_url=website_url,
        isConnectionPageReached=state.get('isConnectionPageReached', False),
        isUsernameFilled=state.get('isUsernameFilled', False),
        isPasswordFilled=state.get('isPasswordFilled', False),
        isLogedIn=state.get('isLogedIn', False),
        isUserProfilPageReached=state.get('isUserProfilPageReached', False),
        isChangeEmailPageReached=state.get('isChangeEmailPageReached', False),
        isActualEmailFilled=state.get('isActualEmailFilled', False),
        isNewEmailFilled=state.get('isNewEmailFilled', False),
        isEmailChanged=state.get('isEmailChanged', False),
    )

    return compiled_prompt