
"use client";

import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { 
  Brain, 
  Key, 
  Users, 
  Settings, 
  Activity, 
  Zap,
  Crown,
  Bot,
  Info
} from "lucide-react";

import AutonomousControlPanel from "@/components/autonomous-control-panel";
import APIKeyManagement from "@/components/api-key-management";
import AutonomousAgentGrid from "@/components/autonomous-agent-grid";

export default function AutonomousPage() {
  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div className="space-y-1">
          <h1 className="text-3xl font-bold tracking-tight flex items-center gap-3">
            <Brain className="h-8 w-8 text-purple-600" />
            Autonomous DSPY Boss
          </h1>
          <p className="text-muted-foreground">
            Fully autonomous task management powered by DSPY signatures. The system makes decisions without manual intervention.
          </p>
        </div>
        
        <div className="flex items-center gap-2">
          <Badge variant="outline" className="flex items-center gap-2">
            <Zap className="h-3 w-3" />
            DSPY-Driven
          </Badge>
          <Badge variant="outline" className="flex items-center gap-2">
            <Activity className="h-3 w-3" />
            Real-time
          </Badge>
        </div>
      </div>

      {/* System Architecture Overview */}
      <Alert className="border-blue-200 bg-blue-50">
        <Info className="h-4 w-4 text-blue-600" />
        <AlertDescription className="text-blue-800">
          <strong>How it works:</strong> DSPY signatures are the core intelligence. They perform pre-processing → 
          autonomous decisions → task execution → error handling → next iteration preparation. The Boss (Agent 0) 
          creates subordinate agents as needed. All decisions are made by AI, not manual controls.
        </AlertDescription>
      </Alert>

      {/* Main Content Tabs */}
      <Tabs defaultValue="control" className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="control" className="flex items-center gap-2">
            <Brain className="h-4 w-4" />
            Control
          </TabsTrigger>
          <TabsTrigger value="agents" className="flex items-center gap-2">
            <Users className="h-4 w-4" />
            Agents
          </TabsTrigger>
          <TabsTrigger value="providers" className="flex items-center gap-2">
            <Key className="h-4 w-4" />
            LLM Providers
          </TabsTrigger>
          <TabsTrigger value="config" className="flex items-center gap-2">
            <Settings className="h-4 w-4" />
            Configuration
          </TabsTrigger>
        </TabsList>

        {/* Autonomous Control Tab */}
        <TabsContent value="control" className="space-y-6">
          <AutonomousControlPanel />
        </TabsContent>

        {/* Agent Hierarchy Tab */}
        <TabsContent value="agents" className="space-y-6">
          <AutonomousAgentGrid />
        </TabsContent>

        {/* LLM Providers Tab */}
        <TabsContent value="providers" className="space-y-6">
          <APIKeyManagement />
        </TabsContent>

        {/* Configuration Tab */}
        <TabsContent value="config" className="space-y-6">
          <div className="grid gap-6">
            {/* Autonomous Settings */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Settings className="h-5 w-5" />
                  Autonomous Operation Settings
                </CardTitle>
                <CardDescription>
                  Configure how the autonomous DSPY engine operates
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Iteration Interval</label>
                      <div className="text-sm text-muted-foreground">Time between autonomous decisions</div>
                      <Badge variant="outline">1.0 seconds</Badge>
                    </div>
                    
                    <div className="space-y-2">
                      <label className="text-sm font-medium">State History Limit</label>
                      <div className="text-sm text-muted-foreground">Number of previous states to remember</div>
                      <Badge variant="outline">100 states</Badge>
                    </div>
                    
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Max Concurrent Agents</label>
                      <div className="text-sm text-muted-foreground">Maximum number of subordinate agents</div>
                      <Badge variant="outline">10 agents</Badge>
                    </div>
                    
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Auto Agent Scaling</label>
                      <div className="text-sm text-muted-foreground">Automatically create/remove agents</div>
                      <Badge variant="outline" className="bg-green-50 text-green-700">
                        Enabled
                      </Badge>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* DSPY Configuration */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Zap className="h-5 w-5" />
                  DSPY Engine Configuration
                </CardTitle>
                <CardDescription>
                  Core DSPY signature settings that power autonomous decisions
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Signature Optimization</label>
                      <div className="text-sm text-muted-foreground">Auto-optimize DSPY signatures</div>
                      <Badge variant="outline" className="bg-blue-50 text-blue-700">
                        Enabled
                      </Badge>
                    </div>
                    
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Retrieval Augmented</label>
                      <div className="text-sm text-muted-foreground">Use retrieval for better decisions</div>
                      <Badge variant="outline" className="bg-blue-50 text-blue-700">
                        Enabled
                      </Badge>
                    </div>
                    
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Error Recovery</label>
                      <div className="text-sm text-muted-foreground">Automatic error recovery and learning</div>
                      <Badge variant="outline" className="bg-green-50 text-green-700">
                        Enabled
                      </Badge>
                    </div>
                    
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Fallback Provider</label>
                      <div className="text-sm text-muted-foreground">Backup LLM provider if primary fails</div>
                      <Badge variant="outline" className="bg-orange-50 text-orange-700">
                        Not configured
                      </Badge>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* System Architecture */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Activity className="h-5 w-5" />
                  System Architecture
                </CardTitle>
                <CardDescription>
                  Understanding the autonomous DSPY-driven architecture
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-6">
                  {/* Iteration Cycle */}
                  <div>
                    <h4 className="font-semibold mb-3">Autonomous Iteration Cycle</h4>
                    <div className="flex flex-col md:flex-row gap-4">
                      {[
                        { phase: "Pre-Processing", icon: "⚡", desc: "Gather and process system data" },
                        { phase: "Boss Decision", icon: "🧠", desc: "DSPY signature makes autonomous decisions" },
                        { phase: "Execution", icon: "⚙️", desc: "Execute tasks via agent assignments" },
                        { phase: "Next Prep", icon: "🔄", desc: "Prepare for next iteration" }
                      ].map((step, index) => (
                        <div key={index} className="flex-1">
                          <div className="border rounded-lg p-3 text-center">
                            <div className="text-2xl mb-2">{step.icon}</div>
                            <div className="font-medium text-sm">{step.phase}</div>
                            <div className="text-xs text-muted-foreground mt-1">{step.desc}</div>
                          </div>
                          {index < 3 && (
                            <div className="hidden md:flex justify-center my-2">
                              <div className="text-muted-foreground">→</div>
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Agent Hierarchy */}
                  <div>
                    <h4 className="font-semibold mb-3">Agent Hierarchy</h4>
                    <div className="space-y-3">
                      <div className="flex items-center gap-3 p-3 border rounded-lg bg-yellow-50">
                        <Crown className="h-6 w-6 text-yellow-600" />
                        <div>
                          <div className="font-medium">Boss (Agent 0)</div>
                          <div className="text-sm text-muted-foreground">
                            Makes all strategic decisions, creates subordinate agents
                          </div>
                        </div>
                      </div>
                      
                      <div className="flex items-center gap-3 p-3 border rounded-lg bg-blue-50">
                        <Bot className="h-6 w-6 text-blue-600" />
                        <div>
                          <div className="font-medium">Subordinates (Agent 1, 2, 3...)</div>
                          <div className="text-sm text-muted-foreground">
                            Execute specific tasks assigned by Boss, numbered sequentially
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
