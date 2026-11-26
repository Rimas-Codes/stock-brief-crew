from typing import List

from crewai import Agent, Crew, LLM, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent

from stck_analyzer.tools.custom_tool import get_stock_data

llm = LLM(model="groq/openai/gpt-oss-120b", temperature=0.1)


@CrewBase
class StockAnalyzerCrew:
    """Stock analyzer crew."""

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def collector(self) -> Agent:
        return Agent(config=self.agents_config['collector'], tools=[get_stock_data], llm=llm, verbose=True)

    @agent
    def summarizer(self) -> Agent:
        return Agent(config=self.agents_config['summarizer'], llm=llm, verbose=True)

    @agent
    def risk_checker(self) -> Agent:
        return Agent(config=self.agents_config['risk_checker'], llm=llm, verbose=True)

    @agent
    def forecaster(self) -> Agent:
        return Agent(config=self.agents_config['forecaster'], llm=llm, verbose=True)

    @agent
    def brief_writer(self) -> Agent:
        return Agent(config=self.agents_config['brief_writer'], llm=llm, verbose=True)

    @task
    def collect_task(self) -> Task:
        return Task(config=self.tasks_config['collect_task'])

    @task
    def summarize_task(self) -> Task:
        return Task(config=self.tasks_config['summarize_task'])

    @task
    def risk_task(self) -> Task:
        return Task(config=self.tasks_config['risk_task'])

    @task
    def forecast_task(self) -> Task:
        return Task(config=self.tasks_config['forecast_task'])

    @task
    def brief_task(self) -> Task:
        return Task(config=self.tasks_config['brief_task'])

    @crew
    def crew(self) -> Crew:
        """Creates the StockAnalyzerCrew crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            max_rpm=10,
            verbose=True,
        )