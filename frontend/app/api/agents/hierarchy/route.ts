
import { NextRequest, NextResponse } from 'next/server';

// Mock agent hierarchy data
const mockAgentHierarchy = {
  boss_agent: {
    agent_id: 0,
    display_name: "Boss (Agent 0)",
    agent_name: "Boss Agent", 
    role: "boss",
    status: "active",
    completed_tasks: 15,
    capabilities: [
      "strategic_decision_making",
      "task_delegation", 
      "system_orchestration",
      "agent_management",
      "priority_assessment",
      "resource_allocation"
    ],
    performance_score: 1.0,
    last_active: new Date().toISOString(),
    current_task: "Monitoring system operations"
  },
  subordinate_agents: [
    {
      agent_id: 1,
      display_name: "Agent 1",
      agent_name: "Trading Agent",
      role: "subordinate",
      status: "idle",
      completed_tasks: 8,
      capabilities: [
        "task_execution",
        "problem_solving", 
        "data_processing",
        "market_analysis",
        "order_execution",
        "risk_management"
      ],
      performance_score: 0.85,
      last_active: new Date(Date.now() - 300000).toISOString(),
      current_task: null
    },
    {
      agent_id: 2,
      display_name: "Agent 2", 
      agent_name: "Analysis Agent",
      role: "subordinate",
      status: "busy",
      completed_tasks: 12,
      capabilities: [
        "task_execution",
        "problem_solving",
        "data_processing", 
        "data_analysis",
        "pattern_recognition",
        "reporting"
      ],
      performance_score: 0.92,
      last_active: new Date().toISOString(),
      current_task: "Analyzing market trends"
    }
  ],
  total_agents: 3,
  next_agent_id: 3,
  hierarchy_stats: {
    active_agents: 2,
    idle_agents: 1,
    busy_agents: 1,
    average_performance: 0.92,
    total_completed_tasks: 35
  }
};

export async function GET(request: NextRequest) {
  try {
    return NextResponse.json({
      success: true,
      data: mockAgentHierarchy
    });
  } catch (error) {
    console.error('Error getting agent hierarchy:', error);
    return NextResponse.json({
      success: false,
      message: "Internal server error"
    }, { status: 500 });
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { action, agent_name, target_count } = body;

    if (action === 'create_agent') {
      if (!agent_name) {
        return NextResponse.json({
          success: false,
          message: "Agent name is required"
        }, { status: 400 });
      }

      // Mock creating new agent
      const newAgent = {
        agent_id: mockAgentHierarchy.next_agent_id,
        display_name: `Agent ${mockAgentHierarchy.next_agent_id}`,
        agent_name: agent_name,
        role: "subordinate",
        status: "idle",
        completed_tasks: 0,
        capabilities: ["task_execution", "problem_solving", "data_processing"],
        performance_score: 0.8,
        last_active: new Date().toISOString(),
        current_task: null
      };

      mockAgentHierarchy.subordinate_agents.push(newAgent);
      mockAgentHierarchy.total_agents++;
      mockAgentHierarchy.next_agent_id++;
      mockAgentHierarchy.hierarchy_stats.idle_agents++;

      return NextResponse.json({
        success: true,
        message: `Created Agent ${newAgent.agent_id} (${agent_name})`,
        data: mockAgentHierarchy
      });
    }

    if (action === 'scale_agents') {
      if (typeof target_count !== 'number') {
        return NextResponse.json({
          success: false,
          message: "Target count must be a number"
        }, { status: 400 });
      }

      const currentSubordinates = mockAgentHierarchy.subordinate_agents.length;
      
      if (target_count > currentSubordinates) {
        // Add agents
        const toAdd = target_count - currentSubordinates;
        for (let i = 0; i < toAdd; i++) {
          const newAgent = {
            agent_id: mockAgentHierarchy.next_agent_id,
            display_name: `Agent ${mockAgentHierarchy.next_agent_id}`,
            agent_name: `Auto Agent ${mockAgentHierarchy.next_agent_id}`,
            role: "subordinate",
            status: "idle",
            completed_tasks: 0,
            capabilities: ["task_execution", "problem_solving", "data_processing"],
            performance_score: 0.8,
            last_active: new Date().toISOString(),
            current_task: null
          };
          
          mockAgentHierarchy.subordinate_agents.push(newAgent);
          mockAgentHierarchy.next_agent_id++;
        }
      } else if (target_count < currentSubordinates) {
        // Remove agents (keep busy ones)
        const toRemove = currentSubordinates - target_count;
        const idleAgents = mockAgentHierarchy.subordinate_agents.filter(a => a.status === 'idle');
        
        for (let i = 0; i < Math.min(toRemove, idleAgents.length); i++) {
          const agentIndex = mockAgentHierarchy.subordinate_agents.findIndex(a => a.agent_id === idleAgents[i].agent_id);
          if (agentIndex > -1) {
            mockAgentHierarchy.subordinate_agents.splice(agentIndex, 1);
          }
        }
      }

      mockAgentHierarchy.total_agents = mockAgentHierarchy.subordinate_agents.length + 1; // +1 for boss
      
      // Update stats
      mockAgentHierarchy.hierarchy_stats = {
        active_agents: mockAgentHierarchy.subordinate_agents.filter(a => a.status === 'active').length + 1,
        idle_agents: mockAgentHierarchy.subordinate_agents.filter(a => a.status === 'idle').length,
        busy_agents: mockAgentHierarchy.subordinate_agents.filter(a => a.status === 'busy').length,
        average_performance: mockAgentHierarchy.subordinate_agents.reduce((sum, a) => sum + a.performance_score, 1.0) / mockAgentHierarchy.total_agents,
        total_completed_tasks: mockAgentHierarchy.subordinate_agents.reduce((sum, a) => sum + a.completed_tasks, 15)
      };

      return NextResponse.json({
        success: true,
        message: `Scaled to ${target_count} subordinate agents`,
        data: mockAgentHierarchy
      });
    }

    return NextResponse.json({
      success: false,
      message: "Invalid action"
    }, { status: 400 });

  } catch (error) {
    console.error('Error managing agent hierarchy:', error);
    return NextResponse.json({
      success: false,
      message: "Internal server error"
    }, { status: 500 });
  }
}
