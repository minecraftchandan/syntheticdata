import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Check, Copy } from "lucide-react";
import { useToast } from "@/hooks/use-toast";

interface CodeBlockProps {
  title: string;
  code: string;
  language?: string;
}

const CodeBlock = ({ title, code, language = "python" }: CodeBlockProps) => {
  const [copied, setCopied] = useState(false);
  const { toast } = useToast();

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      toast({
        title: "Code copied!",
        description: "The code has been copied to your clipboard.",
      });
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      toast({
        title: "Failed to copy",
        description: "Please try copying manually.",
        variant: "destructive",
      });
    }
  };

  return (
    <div className="bg-gradient-card border border-border/50 rounded-lg overflow-hidden shadow-card">
      <div className="flex items-center justify-between px-6 py-4 border-b border-border/50 bg-muted/20">
        <h3 className="font-semibold text-card-foreground flex items-center gap-2">
          <span className="w-3 h-3 bg-destructive rounded-full"></span>
          <span className="w-3 h-3 bg-warning rounded-full"></span>
          <span className="w-3 h-3 bg-success rounded-full"></span>
          <span className="ml-3">{title}</span>
        </h3>
        <Button
          variant="copy"
          size="sm"
          onClick={handleCopy}
          className="h-8"
        >
          {copied ? (
            <Check className="h-4 w-4" />
          ) : (
            <Copy className="h-4 w-4" />
          )}
          {copied ? "Copied!" : "Copy"}
        </Button>
      </div>
      <div className="p-6 overflow-x-auto">
        <pre className="text-sm text-muted-foreground leading-relaxed">
          <code className={`language-${language}`}>
            {code}
          </code>
        </pre>
      </div>
    </div>
  );
};

export default CodeBlock;