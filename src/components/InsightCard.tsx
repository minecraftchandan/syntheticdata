import { ReactNode } from "react";

interface InsightCardProps {
  title: string;
  children: ReactNode;
  type?: "primary" | "accent" | "warning" | "success";
  icon?: string;
}

const InsightCard = ({ title, children, type = "primary", icon }: InsightCardProps) => {
  const typeStyles = {
    primary: "border-primary/30 bg-primary/5",
    accent: "border-accent/30 bg-accent/5", 
    warning: "border-warning/30 bg-warning/5",
    success: "border-success/30 bg-success/5"
  };

  const iconStyles = {
    primary: "text-primary",
    accent: "text-accent",
    warning: "text-warning", 
    success: "text-success"
  };

  return (
    <div className={`rounded-lg border-2 p-6 backdrop-blur-sm transition-all duration-300 hover:scale-[1.02] ${typeStyles[type]}`}>
      <h3 className="flex items-center gap-3 text-lg font-bold mb-4 text-card-foreground">
        {icon && <span className={`text-2xl ${iconStyles[type]}`}>{icon}</span>}
        {title}
      </h3>
      <div className="text-muted-foreground leading-relaxed">
        {children}
      </div>
    </div>
  );
};

export default InsightCard;