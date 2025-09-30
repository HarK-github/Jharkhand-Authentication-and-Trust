"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { cn } from "@/lib/utils"

const nav = [
  { href: "/", label: "Home" },
  { href: "/verify", label: "Verify" },
  { href: "/institution", label: "Institution Portal" },
  { href: "/admin", label: "Admin" },
  { href: "/tech", label: "Technology" },
  { href: "/help", label: "Help" },
]

export function SiteHeader() {
  const pathname = usePathname()
  return (
    <header className="border-b bg-background">
      <div className="mx-auto max-w-6xl px-4 py-3 flex items-center justify-between">
        <Link href="/" className="font-semibold text-lg text-balance">
          Jharkhand Academic Trust (JAT)
        </Link>
        <nav className="flex items-center gap-4">
          {nav.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              aria-current={pathname === item.href ? "page" : undefined}
              className={cn(
                "text-sm px-3 py-2 rounded-md",
                pathname === item.href
                  ? "bg-primary text-primary-foreground"
                  : "text-foreground/80 hover:text-foreground",
              )}
            >
              {item.label}
            </Link>
          ))}
        </nav>
      </div>
    </header>
  )
}
