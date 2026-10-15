"""Rule-based computer and network troubleshooting expert system."""
import streamlit as st

if not st.session_state.get("_embedded_ai_project"):
    st.set_page_config(page_title="Troubleshooting Expert System",page_icon="🛠️",layout="wide")
st.markdown("<style>.block-container{max-width:950px;padding-top:2rem}</style>",unsafe_allow_html=True)
st.title("Rule-Based Troubleshooting Expert System")
st.caption("Select observed symptoms to run transparent forward-chaining rules and inspect their evidence.")
st.warning("Educational troubleshooting aid only. It cannot inspect your device, guarantee a diagnosis, or replace qualified technical support.")
facts=st.multiselect("Observed facts",[
    "device_will_not_power_on","power_light_is_off","battery_may_be_empty",
    "device_powers_on","screen_is_black","external_monitor_works",
    "no_internet","other_devices_have_internet","wifi_is_disconnected",
    "wifi_is_connected","website_fails","other_websites_work",
    "computer_is_slow","disk_space_is_low","many_apps_are_open",
],default=[])

RULES=[
    ({"device_will_not_power_on","power_light_is_off"},"power_issue","Check the power cable, outlet, and adapter."),
    ({"power_issue","battery_may_be_empty"},"possible_empty_battery","Charge the device for at least 30 minutes using a known-good charger."),
    ({"device_powers_on","screen_is_black","external_monitor_works"},"possible_internal_display_issue","The external monitor works, so inspect display brightness, display mode, and internal display hardware."),
    ({"no_internet","other_devices_have_internet","wifi_is_disconnected"},"local_wifi_connection_issue","Reconnect to the Wi-Fi network and confirm the correct network credentials."),
    ({"no_internet","other_devices_have_internet","wifi_is_connected"},"single_device_network_issue","Restart the device's network adapter and verify its IP configuration."),
    ({"website_fails","other_websites_work"},"single_site_issue","The connection works for other sites; check the affected site's status or address."),
    ({"computer_is_slow","disk_space_is_low"},"storage_pressure","Free disk space and review large temporary files."),
    ({"computer_is_slow","many_apps_are_open"},"resource_pressure","Close unused applications and inspect CPU and memory use."),
]
derived=set(facts);fired=[]
changed=True
while changed:
    changed=False
    for required,conclusion,advice in RULES:
        if required<=derived and conclusion not in derived:
            derived.add(conclusion);fired.append((required,conclusion,advice));changed=True
st.subheader("Inferred findings")
if not fired:st.info("No rule matched all required observations. Add relevant facts or consult a technician.")
for required,conclusion,advice in fired:
    with st.expander(conclusion.replace("_"," ").title(),expanded=True):
        st.write(advice)
        st.caption("Rule evidence: "+", ".join(sorted(required)))
st.subheader("Rule engine")
st.write(f"Starting facts: {len(facts)} · rules fired: {len(fired)} · derived conclusions: {len(derived)-len(facts)}")
if st.checkbox("Show all rules"):
    st.dataframe([{"Requires":", ".join(sorted(required)),"Conclusion":conclusion,"Suggested action":advice} for required,conclusion,advice in RULES],hide_index=True,use_container_width=True)
