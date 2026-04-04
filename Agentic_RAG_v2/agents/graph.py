# agents/graph.py
from langgraph.graph import StateGraph, END
from agents.state import AgentState 
from agents.orchestrator_node import OrchestratorAgent
from agents.policy_retriever_node import RegionalPolicyAgent
from agents.quantitative_node import QuantitativeAgent
from agents.developer_node import DeveloperAgent
from agents.critic_node import SynthesisAgent

def build_agentic_workflow(llm_fleet, india_indexes, sg_indexes):
    """
    Compiles the dynamic Multi-Agent State Machine.
    """
    workflow = StateGraph(AgentState)

    # 1. Initialize Agents (Order matters: Define before adding to nodes)
    # Use the specialized LLMs from the fleet for each role
    orchestrator = OrchestratorAgent(llm_fleet["fast_llm"])
    
    # Regional agents take the pre-loaded FAISS/BM25 indexes
    india_agent = RegionalPolicyAgent("India", **india_indexes)
    sg_agent = RegionalPolicyAgent("Singapore", **sg_indexes)
    
    # Reasoning-heavy agents
    quant_agent = QuantitativeAgent(llm_fleet["smart_llm"])
    dev_agent = DeveloperAgent(llm_fleet["writer_llm"])
    synthesizer = SynthesisAgent(llm_fleet["writer_llm"])

    # 2. Add Nodes
    workflow.add_node("Orchestrator", orchestrator)
    workflow.add_node("India_Policy", india_agent)
    workflow.add_node("Singapore_Policy", sg_agent)
    workflow.add_node("Quantitative_Engine", quant_agent)
    workflow.add_node("Developer_Engine", dev_agent)
    workflow.add_node("Synthesizer", synthesizer)

    # 3. The Switchboard (Universal Router)
    def universal_router(state: AgentState):
        # Every agent must return a 'next_speaker' string in their output dict
        return state.get("next_speaker", "end")

    # This map bridges the string returned by an LLM to the physical Node name
    routing_map = {
        "Orchestrator": "Orchestrator",
        "India_Policy": "India_Policy",
        "Singapore_Policy": "Singapore_Policy",
        "Quantitative_Engine": "Quantitative_Engine",
        "Developer_Engine": "Developer_Engine",
        "Synthesizer": "Synthesizer",
        "end": END
    }

    # 4. Define Logic Flow
    workflow.set_entry_point("Orchestrator")

    # Add dynamic edges to all nodes
    # This allows the LLM to decide if it needs a second opinion or more data
    workflow.add_conditional_edges("Orchestrator", universal_router, routing_map)
    workflow.add_conditional_edges("India_Policy", universal_router, routing_map)
    workflow.add_conditional_edges("Singapore_Policy", universal_router, routing_map)
    workflow.add_conditional_edges("Quantitative_Engine", universal_router, routing_map)
    workflow.add_conditional_edges("Developer_Engine", universal_router, routing_map)
    workflow.add_conditional_edges("Synthesizer", universal_router, routing_map)

    return workflow.compile()