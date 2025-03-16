import argparse
import asyncio
import logging
import os
import re
import wandb
from datetime import datetime
from pathlib import Path
from typing import Literal, Dict, Optional

from forecasting_tools import (
    AskNewsSearcher,
    BinaryQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusApi,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    PredictedOptionList,
    PredictionExtractor,
    ReasonedPrediction,
    SmartSearcher,
    ForecastReport,
    clean_indents,
)
import typeguard

# Create logs directory if it doesn't exist
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

# Create a timestamp for this run
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_log_dir = logs_dir / timestamp
run_log_dir.mkdir(exist_ok=True)

# Configure root logger to write to a main log file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(run_log_dir / "main.log"),
        logging.StreamHandler()  # Keep console output
    ]
)
logger = logging.getLogger(__name__)

# Suppress LiteLLM logging
litellm_logger = logging.getLogger("LiteLLM")
litellm_logger.setLevel(logging.WARNING)
litellm_logger.propagate = False

# Dictionary to store question-specific loggers
question_loggers: Dict[str, logging.Logger] = {}

# Initialize wandb
def init_wandb(run_mode: str):
    """Initialize wandb with the appropriate project and run name."""
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if not wandb_api_key:
        logger.warning("WANDB_API_KEY not found in environment variables. WandB logging disabled.")
        return False
    
    project_name = os.getenv("WANDB_PROJECT", "metaculus-forecasting-bot")
    run_name = f"{run_mode}-{timestamp}"
    
    try:
        wandb.init(
            project=project_name,
            name=run_name,
            config={
                "run_mode": run_mode,
                "timestamp": timestamp,
            }
        )
        logger.info(f"WandB initialized with project: {project_name}, run: {run_name}")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize WandB: {e}")
        return False

def extract_question_id(url: str) -> Optional[str]:
    """Extract the question ID from a Metaculus URL."""
    # Try the standard format first
    match = re.search(r'questions/(\d+)', url)
    if match:
        return match.group(1)
    
    # Try alternative formats if needed
    match = re.search(r'/(\d+)/', url)
    if match:
        return match.group(1)
    
    logger.warning(f"Could not extract question ID from URL: {url}")
    return None

def get_question_logger(question: MetaculusQuestion) -> logging.Logger:
    """Get or create a logger for a specific question."""
    question_id = extract_question_id(question.page_url)
    if not question_id:
        logger.warning(f"Could not extract question ID from URL: {question.page_url}")
        return logger  # Fall back to main logger if ID can't be extracted
    
    logger.info(f"Getting logger for question ID: {question_id}, URL: {question.page_url}")
    
    if question_id not in question_loggers:
        # Create a new logger for this question
        q_logger = logging.getLogger(f"question_{question_id}")
        q_logger.setLevel(logging.INFO)
        
        # Create a file handler for this question
        log_file = run_log_dir / f"question_{question_id}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        
        # Add the handler to the logger
        q_logger.addHandler(file_handler)
        q_logger.propagate = False  # Don't propagate to parent logger
        
        # Store the logger
        question_loggers[question_id] = q_logger
        logger.info(f"Created new logger for question ID: {question_id}, log file: {log_file}")
        
        # Log initial question information
        q_logger.info(f"===== QUESTION INFORMATION =====")
        q_logger.info(f"Question ID: {question_id}")
        q_logger.info(f"Question URL: {question.page_url}")
        q_logger.info(f"Question text: {question.question_text}")
        q_logger.info(f"Background info: {question.background_info}")
        q_logger.info(f"Resolution criteria: {question.resolution_criteria}")
        q_logger.info(f"Question type: {question.__class__.__name__}")
        q_logger.info(f"Created at: {timestamp}")
        q_logger.info(f"===============================")
        
        # Log to wandb if initialized
        if wandb.run is not None:
            # Create a wandb Table for text data
            info_table = wandb.Table(columns=["Field", "Value"])
            info_table.add_data("Question ID", question_id)
            info_table.add_data("URL", question.page_url)
            info_table.add_data("Question Text", question.question_text)
            info_table.add_data("Background Info", question.background_info)
            info_table.add_data("Resolution Criteria", question.resolution_criteria)
            info_table.add_data("Question Type", question.__class__.__name__)
            info_table.add_data("Created At", timestamp)
            
            wandb.log({f"question_{question_id}/info": info_table})
    else:
        logger.info(f"Using existing logger for question ID: {question_id}")
    
    return question_loggers[question_id]

class Q1TemplateBot(ForecastBot):
    """
    This is a template bot that uses the forecasting-tools library to simplify bot making.

    The main entry point of this bot is `forecast_on_tournament` in the parent class.
    However generally the flow is:
    - Load questions from Metaculus
    - For each question
        - Execute run_research for research_reports_per_question runs
        - Execute respective run_forecast function for `predictions_per_research_report * research_reports_per_question` runs
        - Aggregate the predictions
        - Submit prediction (if publish_reports_to_metaculus is True)
    - Return a list of ForecastReport objects

    Only the research and forecast functions need to be implemented in ForecastBot subclasses.

    If you end up having trouble with rate limits and want to try a more sophisticated rate limiter try:
    ```
    from forecasting_tools.ai_models.resource_managers.refreshing_bucket_rate_limiter import RefreshingBucketRateLimiter
    rate_limiter = RefreshingBucketRateLimiter(
        capacity=2,
        refresh_rate=1,
    ) # Allows 1 request per second on average with a burst of 2 requests initially. Set this as a class variable
    await self.rate_limiter.wait_till_able_to_acquire_resources(1) # 1 because it's consuming 1 request (use more if you are adding a token limit)
    ```

    Check out https://github.com/Metaculus/forecasting-tools for a full set of features from this package.
    Most notably there is a built in benchmarker that integrates with ForecastBot objects.
    """

    _max_concurrent_questions = (
        10  # Set this to whatever works for your search-provider/ai-model rate limits
    )
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            q_logger = get_question_logger(question)
            research = ""
            if os.getenv("ASKNEWS_CLIENT_ID") and os.getenv("ASKNEWS_SECRET"):
                research = AskNewsSearcher().get_formatted_news(question.question_text)
            elif os.getenv("EXA_API_KEY"):
                research = await self._call_exa_smart_searcher(question.question_text)
            elif os.getenv("PERPLEXITY_API_KEY"):
                research = await self._call_perplexity(question.question_text)
            elif os.getenv("OPENROUTER_API_KEY"):
                research = await self._call_perplexity(question.question_text, use_open_router=True)
            else:
                research = ""
            q_logger.info(f"Found Research for {question.page_url}:\n{research}")
            
            # Log research to wandb
            question_id = extract_question_id(question.page_url)
            if wandb.run is not None and question_id:
                # Create a wandb Table for research text
                research_table = wandb.Table(columns=["Research"])
                research_table.add_data(research)
                wandb.log({
                    f"question_{question_id}/research": research_table
                })
                
            return research

    async def _call_perplexity(self, question: str, use_open_router: bool = False) -> str:
        prompt = clean_indents(
            f"""
            You are an assistant to a superforecaster.
            The superforecaster will give you a question they intend to forecast on.
            To be a great assistant, you generate a concise but detailed rundown of the most relevant news, including if the question would resolve Yes or No based on current information.
            You do not produce forecasts yourself.

            Question:
            {question}
            """
        )
        if use_open_router:
            model_name = "openrouter/perplexity/sonar-reasoning"
        else:
            model_name = "perplexity/sonar-pro" # perplexity/sonar-reasoning and perplexity/sonar are cheaper, but do only 1 search.
        model = GeneralLlm(
            model=model_name,
            temperature=0.1,
        )
        
        # Log model invocation details - using main logger since this isn't attached to a specific question yet
        logger.info(f"Invoking {model_name} for research with prompt: {prompt[:200]}...")
        
        response = await model.invoke(prompt)
        logger.info(f"Received response from {model_name} (length: {len(response)})")
        return response

    async def _call_exa_smart_searcher(self, question: str) -> str:
        """
        SmartSearcher is a custom class that is a wrapper around a search on Exa.ai
        """
        logger.info(f"Initializing SmartSearcher for question: {question[:200]}...")
        
        model = self._get_final_decision_llm()
        logger.info(f"Using model {model.model} for SmartSearcher")
        
        searcher = SmartSearcher(
            model=model,
            temperature=0,
            num_searches_to_run=2,
            num_sites_per_search=10,
        )
        prompt = (
            "You are an assistant to a superforecaster. The superforecaster will give"
            "you a question they intend to forecast on. To be a great assistant, you generate"
            "a concise but detailed rundown of the most relevant news, including if the question"
            "would resolve Yes or No based on current information. You do not produce forecasts yourself."
            f"\n\nThe question is: {question}"
        )  # You can ask the searcher to filter by date, exclude/include a domain, and run specific searches for finding sources vs finding highlights within a source
        
        logger.info(f"Invoking SmartSearcher with prompt: {prompt[:200]}...")
        response = await searcher.invoke(prompt)
        logger.info(f"Received response from SmartSearcher (length: {len(response)})")
        
        return response

    def _get_final_decision_llm(self) -> GeneralLlm:
        model = None
        if os.getenv("OPENROUTER_API_KEY"):
            model = GeneralLlm(model="openrouter/deepseek/deepseek-r1:free", temperature=0.3)
        elif os.getenv("METACULUS_TOKEN"):
            model = GeneralLlm(model="metaculus/gpt-4o", temperature=0.3)
        elif os.getenv("OPENAI_API_KEY"):
            model = GeneralLlm(model="gpt-4o", temperature=0.3)
        elif os.getenv("ANTHROPIC_API_KEY"):
            model = GeneralLlm(model="claude-3-5-sonnet-20241022", temperature=0.3)
        
        else:
            raise ValueError("No API key for final_decision_llm found")
        return model

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Question background:
            {question.background_info}


            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A brief description of a scenario that results in a No outcome.
            (d) A brief description of a scenario that results in a Yes outcome.

            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.

            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )
        q_logger = get_question_logger(question)
        q_logger.info(f"Invoking model for binary forecast with prompt: {prompt[:200]}...")
        model = self._get_final_decision_llm()
        q_logger.info(f"Using model {model.model} for binary forecast")
        
        reasoning = await model.invoke(prompt)
        q_logger.info(f"Received response from model (length: {len(reasoning)})")
        
        q_logger.info(f"Extracting binary prediction from response...")
        try:
            prediction: float = PredictionExtractor.extract_last_percentage_value(
                reasoning, max_prediction=1, min_prediction=0
            )
            q_logger.info(
                f"Forecasted {question.page_url} as {prediction} with reasoning:\n{reasoning}"
            )
            
            # Log to wandb
            question_id = extract_question_id(question.page_url)
            if wandb.run is not None and question_id:
                # Create a wandb Table for reasoning text
                reasoning_table = wandb.Table(columns=["Reasoning"])
                reasoning_table.add_data(reasoning)
                
                wandb.log({
                    f"question_{question_id}/binary_prediction": float(prediction),
                    f"question_{question_id}/reasoning": reasoning_table
                })
                
            return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)
        except Exception as e:
            q_logger.error(f"Error extracting binary prediction: {e}")
            q_logger.error(f"Full reasoning text: {reasoning}")
            
            # Log error to wandb
            question_id = extract_question_id(question.page_url)
            if wandb.run is not None and question_id:
                error_table = wandb.Table(columns=["Error", "Reasoning"])
                error_table.add_data(str(e), reasoning)
                wandb.log({
                    f"question_{question_id}/extraction_error": error_table
                })
            
            # Re-raise the exception to be handled by the caller
            raise

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            The options are: {question.options}


            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A description of an scenario that results in an unexpected outcome.

            You write your rationale remembering that (1) good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time, and (2) good forecasters leave some moderate probability on most options to account for unexpected outcomes.

            The last thing you write is your final probabilities for the N options in this order {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            """
        )
        q_logger = get_question_logger(question)
        q_logger.info(f"Invoking model for multiple choice forecast with prompt: {prompt[:200]}...")
        model = self._get_final_decision_llm()
        q_logger.info(f"Using model {model.model} for multiple choice forecast")
        
        reasoning = await model.invoke(prompt)
        q_logger.info(f"Received response from model (length: {len(reasoning)})")
        
        q_logger.info(f"Extracting multiple choice prediction from response...")
        try:
            prediction: PredictedOptionList = (
                PredictionExtractor.extract_option_list_with_percentage_afterwards(
                    reasoning, question.options
                )
            )
            q_logger.info(
                f"Forecasted {question.page_url} as {prediction} with reasoning:\n{reasoning}"
            )
            
            # Log to wandb
            question_id = extract_question_id(question.page_url)
            if wandb.run is not None and question_id:
                # Convert prediction to dict for easier visualization in wandb
                # Handle PredictedOption objects properly by extracting their values
                prediction_dict = {}
                for i, (option, prob) in enumerate(zip(question.options, prediction)):
                    # Handle the case where prob might be a tuple
                    if isinstance(prob, tuple):
                        prob_value = prob[0] if prob else 0.0
                    else:
                        prob_value = prob
                    prediction_dict[f"option_{i}_{option[:20]}"] = float(prob_value)
                
                # Create a wandb Table for reasoning text
                reasoning_table = wandb.Table(columns=["Reasoning"])
                reasoning_table.add_data(reasoning)
                
                wandb.log({
                    f"question_{question_id}/multiple_choice_prediction": prediction_dict,
                    f"question_{question_id}/reasoning": reasoning_table
                })
                
            return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)
        except Exception as e:
            q_logger.error(f"Error extracting multiple choice prediction: {e}")
            q_logger.error(f"Full reasoning text: {reasoning}")
            
            # Log error to wandb
            question_id = extract_question_id(question.page_url)
            if wandb.run is not None and question_id:
                error_table = wandb.Table(columns=["Error", "Reasoning"])
                error_table.add_data(str(e), reasoning)
                wandb.log({
                    f"question_{question_id}/extraction_error": error_table
                })
            
            # Re-raise the exception to be handled by the caller
            raise

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        q_logger = get_question_logger(question)
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - Please notice the units requested (e.g. whether you represent a number as 1,000,000 or 1m).
            - Never use scientific notation.
            - Always start with a smaller number (more negative if negative) and then increase from there

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) The expectations of experts and markets.
            (e) A brief description of an unexpected scenario that results in a low outcome.
            (f) A brief description of an unexpected scenario that results in a high outcome.

            You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

            The last thing you write is your final answer as:
            "
            Percentile 10: XX
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX
            "
            """
        )
        q_logger.info(f"Invoking model for numeric forecast with prompt: {prompt[:200]}...")
        model = self._get_final_decision_llm()
        q_logger.info(f"Using model {model.model} for numeric forecast")
        
        reasoning = await model.invoke(prompt)
        q_logger.info(f"Received response from model (length: {len(reasoning)})")
        
        q_logger.info(f"Extracting numeric distribution from response...")
        try:
            prediction: NumericDistribution = (
                PredictionExtractor.extract_numeric_distribution_from_list_of_percentile_number_and_probability(
                    reasoning, question
                )
            )
            q_logger.info(
                f"Forecasted {question.page_url} as {prediction.declared_percentiles} with reasoning:\n{reasoning}"
            )
            
            # Log to wandb
            question_id = extract_question_id(question.page_url)
            if wandb.run is not None and question_id:
                # Convert prediction to dict for easier visualization in wandb
                # Handle NumericDistribution objects properly
                percentiles_dict = {}
                
                # Check if declared_percentiles is a list or a dict
                if hasattr(prediction.declared_percentiles, 'items'):
                    # It's a dict
                    for p, v in prediction.declared_percentiles.items():
                        percentiles_dict[f"percentile_{p}"] = float(v)
                else:
                    # It's a list or some other iterable
                    for i, v in enumerate(prediction.declared_percentiles):
                        percentiles_dict[f"percentile_{i}"] = float(v)
                
                # Create a wandb Table for reasoning text
                reasoning_table = wandb.Table(columns=["Reasoning"])
                reasoning_table.add_data(reasoning)
                
                wandb.log({
                    f"question_{question_id}/numeric_prediction": percentiles_dict,
                    f"question_{question_id}/reasoning": reasoning_table
                })
                
            return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)
        except Exception as e:
            q_logger.error(f"Error extracting numeric distribution: {e}")
            q_logger.error(f"Full reasoning text: {reasoning}")
            
            # Log error to wandb
            question_id = extract_question_id(question.page_url)
            if wandb.run is not None and question_id:
                error_table = wandb.Table(columns=["Error", "Reasoning"])
                error_table.add_data(str(e), reasoning)
                wandb.log({
                    f"question_{question_id}/extraction_error": error_table
                })
            
            # Re-raise the exception to be handled by the caller
            raise

    def _create_upper_and_lower_bound_messages(
        self, question: NumericQuestion
    ) -> tuple[str, str]:
        if question.open_upper_bound:
            upper_bound_message = ""
        else:
            upper_bound_message = (
                f"The outcome can not be higher than {question.upper_bound}."
            )
        if question.open_lower_bound:
            lower_bound_message = ""
        else:
            lower_bound_message = (
                f"The outcome can not be lower than {question.lower_bound}."
            )
        return upper_bound_message, lower_bound_message

def summarize_reports(forecast_reports: list[ForecastReport | BaseException]) -> None:
    valid_reports = [
        report for report in forecast_reports if isinstance(report, ForecastReport)
    ]
    exceptions = [
        report for report in forecast_reports if isinstance(report, BaseException)
    ]
    minor_exceptions = [
        report.errors for report in valid_reports if report.errors
    ]

    logger.info(f"Summarizing {len(valid_reports)} valid reports, {len(exceptions)} exceptions")
    
    # Log summary to wandb
    if wandb.run is not None:
        wandb.log({
            "summary/valid_reports": len(valid_reports),
            "summary/exceptions": len(exceptions),
            "summary/minor_exceptions": len(minor_exceptions)
        })
    
    for report in valid_reports:
        question_id = extract_question_id(report.question.page_url)
        logger.info(f"Processing report for question ID: {question_id}, URL: {report.question.page_url}")
        
        if question_id:
            if question_id in question_loggers:
                q_logger = question_loggers[question_id]
                logger.info(f"Found logger for question ID: {question_id}")
            else:
                logger.warning(f"No logger found for question ID: {question_id}, falling back to main logger")
                q_logger = logger
        else:
            logger.warning(f"Could not extract question ID from URL: {report.question.page_url}")
            q_logger = logger
        
        question_summary = clean_indents(f"""
            URL: {report.question.page_url}
            Errors: {report.errors}
            Summary:
            {report.summary}
            ---------------------------------------------------------
        """)
        q_logger.info(question_summary)
        logger.info(f"Completed forecast for question {question_id}: {report.question.page_url}")
        
        # Log final summary to wandb
        if wandb.run is not None and question_id:
            # Create a wandb Table for summary text
            summary_table = wandb.Table(columns=["Summary"])
            summary_table.add_data(report.summary)
            
            # Create a wandb Table for errors text if any
            if report.errors:
                errors_table = wandb.Table(columns=["Errors"])
                errors_table.add_data(str(report.errors))
                wandb.log({
                    f"question_{question_id}/final_summary": summary_table,
                    f"question_{question_id}/errors": errors_table
                })
            else:
                wandb.log({
                    f"question_{question_id}/final_summary": summary_table
                })

    if exceptions:
        error_msg = f"{len(exceptions)} errors occurred while forecasting: {exceptions}"
        logger.error(error_msg)
        if wandb.run is not None:
            wandb.log({"errors": error_msg})
        raise RuntimeError(error_msg)
    
    if minor_exceptions:
        minor_error_msg = f"{len(minor_exceptions)} minor exceptions occurred while forecasting: {minor_exceptions}"
        logger.error(minor_error_msg)
        if wandb.run is not None:
            wandb.log({"minor_errors": minor_error_msg})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Q1TemplateBot forecasting system"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "quarterly_cup", "test_questions"],
        default="tournament",
        help="Specify the run mode (default: tournament)",
    )
    parser.add_argument(
        "--predictions_per_report",
        type=int,
        default=5,
        help="Number of predictions per research report",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=5,
        help="Maximum number of retries for failed API calls",
    )
    parser.add_argument(
        "--retry_delay",
        type=int,
        default=3,
        help="Delay between retries in seconds",
    )
    args = parser.parse_args()
    run_mode: Literal["tournament", "quarterly_cup", "test_questions"] = args.mode
    assert run_mode in [
        "tournament",
        "quarterly_cup",
        "test_questions",
    ], "Invalid run mode"

    logger.info(f"Starting forecasting run in {run_mode} mode at {timestamp}")
    logger.info(f"Logs will be saved to {run_log_dir}")
    logger.info(f"Individual question logs will be created in {run_log_dir} as question_[ID].log files")
    
    # Initialize wandb
    wandb_enabled = init_wandb(run_mode)
    if wandb_enabled:
        logger.info("WandB logging enabled")
    else:
        logger.warning("WandB logging disabled")

    template_bot = Q1TemplateBot(
        research_reports_per_question=1,
        predictions_per_research_report=args.predictions_per_report,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=True,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
    )

    if run_mode == "tournament":
        Q4_2024_AI_BENCHMARKING_ID = 32506
        Q1_2025_AI_BENCHMARKING_ID = 32627
        forecast_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                Q1_2025_AI_BENCHMARKING_ID, return_exceptions=True
            )
        )
    elif run_mode == "quarterly_cup":
        # The quarterly cup is a good way to test the bot's performance on regularly open questions. You can also use AXC_2025_TOURNAMENT_ID = 32564
        Q1_2025_QUARTERLY_CUP_ID = 32630
        template_bot.skip_previously_forecasted_questions = False
        forecast_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                Q1_2025_QUARTERLY_CUP_ID, return_exceptions=True
            )
        )
    elif run_mode == "test_questions":
        # Example questions are a good way to test the bot's performance on a single question
        EXAMPLE_QUESTIONS = [
            "https://www.metaculus.com/questions/578/human-extinction-by-2100/",  # Human Extinction - Binary
            "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",  # Age of Oldest Human - Numeric
            "https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",  # Number of New Leading AI Labs - Multiple Choice
        ]
        template_bot.skip_previously_forecasted_questions = False
        questions = [
            MetaculusApi.get_question_by_url(question_url)
            for question_url in EXAMPLE_QUESTIONS
        ]
        forecast_reports = asyncio.run(
            template_bot.forecast_questions(
                questions, return_exceptions=True
            )
        )
    forecast_reports = typeguard.check_type(forecast_reports, list[ForecastReport | BaseException])
    summarize_reports(forecast_reports)

    # Add final log message to both main and all question loggers
    logger.info(f"Completed forecasting run in {run_mode} mode")
    for q_logger in question_loggers.values():
        q_logger.info(f"Completed forecasting run in {run_mode} mode")
    
    # Finish wandb run
    if wandb.run is not None:
        wandb.log({"status": "completed"})
        wandb.finish()

