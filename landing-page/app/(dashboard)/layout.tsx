import { redirect } from "next/navigation";
import { createClient } from "@/lib/supabase/server";
import { DashboardSidebar } from "@/components/dashboard/Sidebar";

export default async function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();

  if (!user) redirect("/auth/login");

  const { data: profile } = await supabase
    .from("profiles")
    .select("tenant")
    .eq("id", user.id)
    .single();

  const tenant = profile?.tenant ?? "default";

  return (
    <div className="flex flex-col md:flex-row h-screen overflow-hidden print:h-auto print:overflow-visible" style={{ background: "#F5F5F7" }}>
      <DashboardSidebar tenant={tenant} userEmail={user.email ?? ""} />
      <main className="flex-1 overflow-y-auto print:overflow-visible">
        {children}
      </main>
    </div>
  );
}
