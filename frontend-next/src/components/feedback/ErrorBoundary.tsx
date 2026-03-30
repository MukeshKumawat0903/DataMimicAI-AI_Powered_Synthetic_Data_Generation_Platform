"use client";

import React from "react";
import { useWorkspaceStore } from "@/lib/store/workspace";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { AlertTriangle, RefreshCw, Home } from "lucide-react";

interface State {
  hasError: boolean;
  error: Error | null;
}

interface Props {
  children: React.ReactNode;
  fallbackTitle?: string;
}

export class ErrorBoundary extends React.Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  handleRetry = () => {
    this.setState({ hasError: false, error: null });
  };

  render() {
    if (!this.state.hasError) {
      return this.props.children;
    }
    return (
      <ErrorFallback
        error={this.state.error}
        onRetry={this.handleRetry}
        title={this.props.fallbackTitle}
      />
    );
  }
}

function ErrorFallback({
  error,
  onRetry,
  title,
}: {
  error: Error | null;
  onRetry: () => void;
  title?: string;
}) {
  const reset = useWorkspaceStore((s) => s.reset);

  function handleReset() {
    reset();
    window.location.replace("/upload");
  }

  return (
    <div className="flex items-center justify-center min-h-40 p-4">
      <Card className="w-full max-w-md border-destructive/50">
        <CardHeader className="flex flex-row items-center gap-2 pb-2">
          <AlertTriangle className="h-5 w-5 text-destructive" />
          <CardTitle className="text-base">
            {title ?? "Something went wrong"}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          {error && (
            <p className="text-sm text-muted-foreground font-mono break-all">
              {error.message}
            </p>
          )}
          <div className="flex gap-2">
            <Button variant="outline" size="sm" onClick={onRetry} className="gap-1">
              <RefreshCw className="h-3.5 w-3.5" />
              Retry
            </Button>
            <Button variant="ghost" size="sm" onClick={handleReset} className="gap-1">
              <Home className="h-3.5 w-3.5" />
              Reset session
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
