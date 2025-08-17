
import Header from '@/components/header';
import BossStateManager from '@/components/boss-state-manager';

export default function BossStatePage() {
  return (
    <div className="min-h-screen bg-gray-50">
      <Header 
        title="Boss State Management" 
        description="Monitor and control the DSPY Boss state machine and system operations"
      />
      <div className="p-6">
        <BossStateManager />
      </div>
    </div>
  );
}
