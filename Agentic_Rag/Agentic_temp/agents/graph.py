from langgraph.graph import StateGraph, END
from agents.state import AgenticRAGState
from agents.orchestrator_node import OrchestratorAgent
from agents.policy_retriever_node import RegionalPolicyAgent
from agents.quantitative_node import QuantitativeAgent
from agents.developer_node import DeveloperAgent
from agents.critic_node import SynthesisAgent

def build_agentic_workflow(llm, india_indexes, sg_indexes):
    """
    Compiles the Multi-Agent State Machine for cross-border financial RAG.
    """
    # 1. Initialize State Graph
    workflow = StateGraph(AgenticRAGState)

    # 2. Instantiate Agents
    orchestrator = OrchestratorAgent(llm)
    india_agent = RegionalPolicyAgent("India", **india_indexes)
    sg_agent = RegionalPolicyAgent("Singapore", **sg_indexes)
    quant_agent = QuantitativeAgent(llm)
    dev_agent = DeveloperAgent(llm)
    synthesizer = SynthesisAgent(llm)

    # 3. Add Nodes to Graph
    workflow.add_node("Orchestrator", orchestrator)
    workflow.add_node("India_Policy", india_agent)
    workflow.add_node("Singapore_Policy", sg_agent)
    workflow.add_node("Quantitative_Engine", quant_agent)
    workflow.add_node("Developer_Engine", dev_agent)
    workflow.add_node("Synthesizer", synthesizer)

    # 4. Define Routing Logic
    workflow.set_entry_point("Orchestrator")

    def route_from_orchestrator(state: AgenticRAGState):
        """Reads the hidden routing_decisions dict to trigger parallel edges."""
        decisions = state.get("routing_decisions", {})
        destinations = []
        if decisions.get("India_Policy_Agent"):
            destinations.append("India_Policy")
        if decisions.get("Singapore_Policy_Agent"):
            destinations.append("Singapore_Policy")
        if decisions.get("Developer_Agent"):
            destinations.append("Developer_Engine")
        
        # Immediate math bypass
        if not destinations and decisions.get("Quantitative_Agent"):
            return ["Quantitative_Engine"]
            
        # Fallback
        if not destinations:
            return ["India_Policy", "Singapore_Policy"]
            
        return destinations

    # Orchestrator conditionally routes to the appropriate agents
    workflow.add_conditional_edges(
        "Orchestrator",
        route_from_orchestrator,
        {
            "India_Policy": "India_Policy",
            "Singapore_Policy": "Singapore_Policy",
            "Quantitative_Engine": "Quantitative_Engine",
            "Developer_Engine": "Developer_Engine"
        }
    )

    # 5. Define Sequential Flow
    # After Policy agents finish, they BOTH flow into the Quantitative Engine
    workflow.add_edge("India_Policy", "Quantitative_Engine")
    workflow.add_edge("Singapore_Policy", "Quantitative_Engine")
    
    # After math or dev tasks are executed, Synthesizer drafts the final response
    workflow.add_edge("Quantitative_Engine", "Synthesizer")
    workflow.add_edge("Developer_Engine", "Synthesizer")
    workflow.add_edge("Synthesizer", END)

    # Compile the graph into a runnable application
    return workflow.compile()