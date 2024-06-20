import type { Metadata } from "next";
import { Inter, IBM_Plex_Serif } from "next/font/google";
import React from "react";

// copied primary layout but removed the metadata and fonts
// this special layout file will be used for the main components - home, dashboard, transactions pages
// everything that uses the sidebar
// we then have a third layout file in (auth) folder 
export default function RootLayout({
    children,
}: Readonly<{
    children: React.ReactNode;
}>) {
    return (
        <main>
            SIDEBAR
            {children}
        </main>
    );
}
