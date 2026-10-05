"use client";

import { API_V1, customInstance } from "@/shared/lib/api/custom-instance";
import type {
  ChemCellarImportResponse,
  ChemCellarProtocolResponse,
  ChemCellarRunResponse,
} from "@/shared/lib/api/model";
import { STALE_TIME } from "@/shared/lib/query-defaults";
import { useMutation, useQuery } from "@tanstack/react-query";

const KEY = ["chemcellar"] as const;

export function useChemCellarProtocols() {
  return useQuery({
    queryKey: [...KEY, "protocols"],
    queryFn: () =>
      customInstance<ChemCellarProtocolResponse[]>({
        url: `${API_V1}/chemcellar/protocols`,
        method: "GET",
      }),
    staleTime: STALE_TIME.MEDIUM,
    retry: false,
  });
}

export function useChemCellarRuns(protocolId: string) {
  return useQuery({
    queryKey: [...KEY, "protocols", protocolId, "runs"],
    queryFn: () =>
      customInstance<ChemCellarRunResponse[]>({
        url: `${API_V1}/chemcellar/protocols/${protocolId}/runs`,
        method: "GET",
      }),
    enabled: protocolId !== "",
    staleTime: STALE_TIME.SHORT,
  });
}

export function useImportChemCellarRun() {
  return useMutation({
    mutationFn: (runId: string) =>
      customInstance<ChemCellarImportResponse>({
        url: `${API_V1}/chemcellar/runs/${runId}/import`,
        method: "POST",
      }),
  });
}
