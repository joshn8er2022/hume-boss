
import { NextRequest, NextResponse } from 'next/server';

// Mock boss connection
const mockBossConnection = {
  stop_autonomous_mode: async () => {
    return true;
  },
  get_autonomous_status: async () => {
    return {
      is_autonomous: false,
      iteration_count: 0,
      current_phase: null,
      current_iteration_id: null,
      system_state: { autonomous_mode: false },
      recent_decisions: [],
      agent_hierarchy: {
        boss_agent: { agent_id: 0, agent_name: "Boss Agent", status: "idle" },
        subordinate_agents: [],
        total_agents: 1
      },
      llm_providers: { active_provider: null }
    };
  }
};

export async function POST(request: NextRequest) {
  try {
    await mockBossConnection.stop_autonomous_mode();
    const status = await mockBossConnection.get_autonomous_status();
    
    return NextResponse.json({
      success: true,
      message: "Autonomous mode stopped successfully",
      status: status
    });
  } catch (error) {
    console.error('Error stopping autonomous mode:', error);
    return NextResponse.json({
      success: false,
      message: "Internal server error"
    }, { status: 500 });
  }
}
