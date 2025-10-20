"""
LangChain-powered structured outputs for environmental consulting.
This replaces the manual validation with LangChain's PydanticOutputParser.
"""

import logging
from typing import Dict, Any, Optional, List, Union, Literal
from datetime import datetime
from pydantic import BaseModel, Field, validator, model_validator

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from .models import Citation, ComplianceReport, RiskAssessment, TechnicalAnalysis

logger = logging.getLogger(__name__)

class LangChainStructuredOutputs:
    """LangChain-powered structured output generator."""
    
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.1):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            streaming=True
        )
        self._setup_parsers()
        self._setup_prompts()
    
    def _setup_parsers(self):
        """Initialize Pydantic output parsers."""
        self.compliance_parser = PydanticOutputParser(pydantic_object=ComplianceReport)
        self.risk_parser = PydanticOutputParser(pydantic_object=RiskAssessment)
        self.technical_parser = PydanticOutputParser(pydantic_object=TechnicalAnalysis)
        self.citation_parser = PydanticOutputParser(pydantic_object=Citation)
    
    def _setup_prompts(self):
        """Setup LangChain prompts with format instructions."""
        
        # Compliance Report Prompt
        self.compliance_prompt = ChatPromptTemplate.from_template("""
You are an expert environmental consultant specializing in Chilean mining regulations.

Analyze the following query and generate a comprehensive compliance report:

Query: {query}
Context: {context}

{format_instructions}

Focus on:
- Chilean environmental regulations (SEA, SMA, DGA, etc.)
- Specific compliance requirements
- Risk assessment
- Actionable recommendations
- Regulatory citations

Ensure all findings are specific and actionable.
""")
        
        # Risk Assessment Prompt
        self.risk_prompt = ChatPromptTemplate.from_template("""
You are an expert environmental risk assessor for mining projects in Chile.

Evaluate the environmental risks for the following query:

Query: {query}
Context: {context}

{format_instructions}

Consider:
- Environmental impact categories
- Probability and impact assessment
- Mitigation strategies
- Monitoring requirements
- Chilean regulatory context

Provide specific, measurable risk factors and mitigation strategies.
""")
        
        # Technical Analysis Prompt
        self.technical_prompt = ChatPromptTemplate.from_template("""
You are a senior environmental engineer specializing in mining projects.

Provide a comprehensive technical analysis for:

Query: {query}
Context: {context}

{format_instructions}

Include:
- Technical specifications
- Engineering solutions
- Performance metrics
- Implementation timeline
- Cost considerations
- Chilean technical standards

Ensure technical accuracy and practical applicability.
""")
    
    def generate_compliance_report(self, query: str, context: str = "") -> ComplianceReport:
        """Generate structured compliance report using LangChain."""
        try:
            # Add format instructions to prompt
            prompt_with_format = self.compliance_prompt.partial(
                format_instructions=self.compliance_parser.get_format_instructions()
            )
            
            # Create chain
            chain = prompt_with_format | self.llm | self.compliance_parser
            
            # Execute
            result = chain.invoke({
                "query": query,
                "context": context
            })
            
            logger.info(f"Generated compliance report: {result.report_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error generating compliance report: {e}")
            # Fallback to manual creation
            return ComplianceReport(
                query=query,
                compliance_status="requires_review",
                risk_level="medium",
                confidence_score=0.5,
                findings=["Error in automated analysis"],
                recommendations=["Manual review required"]
            )
    
    def generate_risk_assessment(self, query: str, context: str = "") -> RiskAssessment:
        """Generate structured risk assessment using LangChain."""
        try:
            prompt_with_format = self.risk_prompt.partial(
                format_instructions=self.risk_parser.get_format_instructions()
            )
            
            chain = prompt_with_format | self.llm | self.risk_parser
            
            result = chain.invoke({
                "query": query,
                "context": context
            })
            
            logger.info(f"Generated risk assessment: {result.assessment_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error generating risk assessment: {e}")
            return RiskAssessment(
                query=query,
                risk_category="environmental",
                risk_level="medium",
                probability=0.5,
                impact=0.5,
                risk_score=0.5,
                confidence_score=0.5,
                risk_factors=["Error in automated analysis"],
                mitigation_strategies=["Manual review required"]
            )
    
    def generate_technical_analysis(self, query: str, context: str = "") -> TechnicalAnalysis:
        """Generate structured technical analysis using LangChain."""
        try:
            prompt_with_format = self.technical_prompt.partial(
                format_instructions=self.technical_parser.get_format_instructions()
            )
            
            chain = prompt_with_format | self.llm | self.technical_parser
            
            result = chain.invoke({
                "query": query,
                "context": context
            })
            
            logger.info(f"Generated technical analysis: {result.analysis_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error generating technical analysis: {e}")
            return TechnicalAnalysis(
                query=query,
                analysis_type="general",
                complexity_level="medium",
                confidence_score=0.5,
                key_findings=["Error in automated analysis"],
                recommendations=["Manual review required"]
            )

# Global instance
langchain_structured_outputs = LangChainStructuredOutputs()
