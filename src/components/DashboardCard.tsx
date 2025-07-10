import { ReactNode } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface DashboardCardProps {
  title: string;
  children: ReactNode;
  className?: string;
  icon?: string;
  glowEffect?: boolean;
}

const DashboardCard = ({ title, children, className = "", icon, glowEffect = false }: DashboardCardProps) => {
  return (
    <Card className={`bg-gradient-card border-border/50 backdrop-blur-sm ${glowEffect ? 'shadow-card hover:shadow-glow' : 'shadow-card'} transition-all duration-300 hover:scale-[1.02] ${className}`}>
      <CardHeader className="pb-4">
        <CardTitle className="flex items-center gap-3 text-xl font-bold text-card-foreground">
          {icon && <span className="text-2xl">{icon}</span>}
          {title}
        </CardTitle>
      </CardHeader>
      <CardContent>
        {children}
      </CardContent>
    </Card>
  );
};

export default DashboardCard;