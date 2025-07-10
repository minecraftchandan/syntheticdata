import { Button } from "@/components/ui/button";

const HeroSection = () => {
  return (
    <section className="relative overflow-hidden bg-gradient-hero py-24 px-6">
      <div className="absolute inset-0 opacity-30">
        <div className="w-full h-full" style={{
          backgroundImage: `url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.03'%3E%3Ccircle cx='30' cy='30' r='4'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")`
        }} />
      </div>
      
      <div className="relative max-w-6xl mx-auto text-center">
        <div className="inline-flex items-center px-4 py-2 rounded-full bg-primary/10 border border-primary/20 text-primary mb-8">
          <span className="w-2 h-2 bg-primary rounded-full mr-2 animate-pulse"></span>
           Gaming Survey Analytics
        </div>
        
        <h1 className="text-5xl lg:text-7xl font-bold mb-6 bg-gradient-primary bg-clip-text text-transparent leading-tight">
          Gaming Preferences & 
          <br />
          Player Profiles Report
        </h1>
        
        <p className="text-xl text-muted-foreground mb-8 max-w-2xl mx-auto leading-relaxed">
          Analaysis of gaming survey from a public access Dataset
        </p>
        <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
          <a href="https://www.kaggle.com/datasets/divyansh22/online-gaming-anxiety-data/data"target="_blank"rel="noopener noreferrer">
           <Button variant="hero" size="lg" className="min-w-48">
             🔗 Dataset Source
           </Button>
           </a>
        </div>        
        <div className="mt-12 text-sm text-muted-foreground">
          <p>Student ID: <span className="text-primary font-mono">21MIC7040</span> | 
             Made-By: <span className="text-primary font-semibold">Chandan Sathvik</span></p>
        </div>
      </div>
      <div className="absolute -bottom-32 -right-32 w-64 h-64 bg-primary/20 rounded-full blur-3xl"></div>
      <div className="absolute -top-32 -left-32 w-64 h-64 bg-accent/20 rounded-full blur-3xl"></div>
    </section>
  );
};

export default HeroSection;