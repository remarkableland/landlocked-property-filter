import streamlit as st
import pandas as pd
import math
from datetime import datetime

st.set_page_config(page_title="Landlocked Property Filter", page_icon="🏞️", layout="wide")

st.title("🏞️ Landlocked Property Filter")
st.markdown("Filter landlocked properties and prepare for mailing list output")

# Define columns to delete (from close-input-file spec)
COLUMNS_TO_DELETE = [
    'OWNER_NAME_STD', 'OWNER_TYPE', 'OWNER_OCCUPIED', 'ASSR_LINK_APN1',
    'PROP_ADDRESS', 'PROP_CITY', 'PROP_STATE', 'PROP_ZIP',
    'LAND_SQFT', 'UNITS_NUMBER', 'CENSUS_BLOCK_GROUP', '_SIMPLIFIED'
]

# Define column renames (from close-input-file spec)
COLUMN_RENAMES = {
    'OWNER_NAME_1': 'NAME',
    'OWNER_1_FIRST': 'FIRST NAME',
    'OWNER_1_LAST': 'LAST NAME',
    'OWNER_ADDRESS': 'ADDRESS',
    'OWNER_CITY': 'CITY',
    'OWNER_STATE': 'address_1_state',
    'OWNER_ZIP': 'ZIP/POSTAL CODE',
    'SITE_STATE': 'custom.State'
}

# Column mapping from Landlocked to Property Search schema
LANDLOCKED_TO_PROPERTY_SEARCH = {
    'OWNER_NAME_1': 'Owner Name(s)',
    'OWNER_NAME_2': 'Owner 2 Full Name',
    'APN': 'APN',
    'SITE_ADDR': 'Parcel Address',
    'SITE_CITY': 'City',
    'SITE_ZIP': 'ZIP',
    'DATE_TRANSFER': 'Last Sale Date',
    'VAL_TRANSFER': 'Last Sale Price',
    'BUILDING_SQFT': 'Structure Sq Ft',
    'ACREAGE': 'Acreage',
    'AGGR_ACREAGE': 'Acreage',  # Same value for unmatched
    'YR_BLT': 'Structure Year Built',
    'ZONING': 'Zoning',
    'USE_CODE_STD_DESC': 'Land Use Description',
    'APPRAISE_VAL': 'Total Parcel Value',
    'VAL_MARKET': 'Market Total Parcel Value',
    'ALTERNATE_APN': 'Parcel Alt APN',
    'COUNTY': 'County',
    'LAST_LOAN_VALUE': 'Mortgage Amount',
    'LATITUDE': 'Latitude',
    'LEGAL_1': 'Legal Description',
    'LONGITUDE': 'Longitude',
    'OWNER_ADDRESS': 'Mail Address',
    'OWNER_CITY': 'Mail City',
    'OWNER_STATE': 'Mail State',
    'OWNER_ZIP': 'Mail ZIP',
    'SELLER_NAME': 'Previous Owner(s)',
    'SITE_STATE': 'State',
    'OWNER_1_FIRST': 'Owner 1 First Name',
    'OWNER_1_LAST': 'Owner 1 Last Name',
}

# Columns that will be empty for unmatched records
EMPTY_COLUMNS = [
    'AGGR_GROUP', 'AGGR_LOT_COUNT', 'USE_CODE_STD_CTGR_DESC',
    'LAST_LOAN_DATE_RECORDING', 'LAST_LOAN_DUE_DATE', 'LEGAL_2', 'LEGAL_3',
    'PROP_TYPE', 'PROP_TYPE_CTGR', 'RANGE', 'SECTION', 'TOWNSHIP', 'CENSUS_TRACT'
]


def haversine_distance_feet(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in feet using Haversine formula"""
    R = 20902231  # Earth radius in feet
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c


def apply_title_case(df):
    """Apply title case to all text columns except state columns"""
    state_columns = ['PROP_STATE', 'SITE_STATE', 'OWNER_STATE', 'address_1_state', 'custom.State']

    for col in df.columns:
        if df[col].dtype == 'object':
            if col in state_columns:
                df[col] = df[col].str.upper()
            else:
                df[col] = df[col].str.title()

    return df


def map_landlocked_to_property_schema(landlocked_df, property_columns):
    """Map unmatched landlocked records to Property Search schema"""
    mapped_data = []

    for _, row in landlocked_df.iterrows():
        new_row = {}

        # Map columns that have equivalents
        for ps_col, ll_col in LANDLOCKED_TO_PROPERTY_SEARCH.items():
            if ll_col in row.index:
                new_row[ps_col] = row[ll_col]
            else:
                new_row[ps_col] = ''

        # Set empty columns
        for col in EMPTY_COLUMNS:
            new_row[col] = ''

        # Ensure all Property Search columns exist
        for col in property_columns:
            if col not in new_row:
                new_row[col] = ''

        mapped_data.append(new_row)

    return pd.DataFrame(mapped_data, columns=property_columns)


def check_owner_has_nearby_property(owner_name, parcel_lat, parcel_lon, all_properties_df, distance_threshold_feet=5280):
    """Check if owner has another property within the distance threshold"""
    if not owner_name or pd.isna(owner_name):
        return False

    # Find all properties owned by this owner
    owner_properties = all_properties_df[
        all_properties_df['OWNER_NAME_1'].str.upper().str.strip() == owner_name.upper().strip()
    ]

    if len(owner_properties) <= 1:
        return False

    # Check if any other property is within the distance threshold
    for _, prop in owner_properties.iterrows():
        try:
            prop_lat = float(prop['LATITUDE'])
            prop_lon = float(prop['LONGITUDE'])

            # Skip if same coordinates (same parcel)
            if abs(prop_lat - parcel_lat) < 0.0001 and abs(prop_lon - parcel_lon) < 0.0001:
                continue

            distance = haversine_distance_feet(parcel_lat, parcel_lon, prop_lat, prop_lon)

            if distance <= distance_threshold_feet:
                return True
        except (ValueError, TypeError):
            continue

    return False


# File Upload Section
st.subheader("📁 Upload Files")

col1, col2 = st.columns(2)

with col1:
    property_file = st.file_uploader(
        "Property Search CSV (Primary Data Source)",
        type="csv",
        help="Upload the county-wide Property Search export"
    )

with col2:
    landlocked_file = st.file_uploader(
        "Landlocked CSV (Filtered List)",
        type="csv",
        help="Upload the landlocked properties export"
    )

# Configuration Section
st.subheader("⚙️ Configuration")

col1, col2, col3 = st.columns(3)

with col1:
    num_codes = st.number_input(
        "Number of Mail_CallRail codes",
        min_value=1,
        max_value=10,
        value=1,
        step=1
    )

with col2:
    lead_type = st.text_input("Lead_Type", value="Landlocked")

with col3:
    mail_type = st.text_input("Mail_Type", value="Neutral Postcard")

# Dynamic code inputs
st.markdown("**Enter Mail_CallRail codes:**")
codes = []
default_codes = ["817-674-1490_Landlocked10_Postcard"]
code_cols = st.columns(min(num_codes, 4))
for i in range(num_codes):
    col_idx = i % 4
    with code_cols[col_idx]:
        default_value = default_codes[i] if i < len(default_codes) else ""
        code = st.text_input(f"Code {i+1}", key=f"code_{i}", value=default_value)
        codes.append(code)

# Validation
config_complete = (
    property_file is not None and
    landlocked_file is not None and
    lead_type and
    mail_type and
    all(codes)
)

if not config_complete:
    st.warning("⚠️ Please upload both files and complete all configuration fields")

# Process Button
if config_complete:
    if st.button("🚀 Process Files", type="primary"):
        try:
            with st.spinner("Processing..."):
                # Step 1: Load files
                st.write("**Step 1:** Loading files...")
                property_df = pd.read_csv(property_file)
                landlocked_df = pd.read_csv(landlocked_file)

                st.success(f"✅ Loaded Property Search: {len(property_df):,} records")
                st.success(f"✅ Loaded Landlocked: {len(landlocked_df):,} records")

                # Step 2: Match APNs
                st.write("**Step 2:** Matching APNs...")
                landlocked_apns = set(landlocked_df['APN'].str.strip())

                matched_df = property_df[property_df['APN'].str.strip().isin(landlocked_apns)].copy()
                matched_apns = set(matched_df['APN'].str.strip())

                unmatched_landlocked_df = landlocked_df[~landlocked_df['APN'].str.strip().isin(matched_apns)].copy()

                st.success(f"✅ Matched: {len(matched_df):,} records")
                st.info(f"ℹ️ Unmatched landlocked records: {len(unmatched_landlocked_df):,}")

                # Step 3: Filter by proximity (1 mile = 5280 feet)
                st.write("**Step 3:** Filtering owners with nearby properties (within 1 mile)...")

                records_to_keep = []
                excluded_count = 0

                progress_bar = st.progress(0)
                total = len(matched_df)

                for idx, (_, row) in enumerate(matched_df.iterrows()):
                    try:
                        lat = float(row['LATITUDE'])
                        lon = float(row['LONGITUDE'])
                        owner = row['OWNER_NAME_1']

                        has_nearby = check_owner_has_nearby_property(
                            owner, lat, lon, property_df, distance_threshold_feet=5280
                        )

                        if not has_nearby:
                            records_to_keep.append(row)
                        else:
                            excluded_count += 1
                    except (ValueError, TypeError):
                        # If we can't parse coordinates, keep the record
                        records_to_keep.append(row)

                    progress_bar.progress((idx + 1) / total)

                filtered_matched_df = pd.DataFrame(records_to_keep)

                st.success(f"✅ Excluded {excluded_count:,} records (owner has property within 1 mile)")
                st.success(f"✅ Remaining matched records: {len(filtered_matched_df):,}")

                # Step 4: Map unmatched landlocked records to Property Search schema
                st.write("**Step 4:** Mapping unmatched records to Property Search schema...")

                if len(unmatched_landlocked_df) > 0:
                    mapped_unmatched_df = map_landlocked_to_property_schema(
                        unmatched_landlocked_df,
                        property_df.columns.tolist()
                    )
                    st.success(f"✅ Mapped {len(mapped_unmatched_df):,} unmatched records")
                else:
                    mapped_unmatched_df = pd.DataFrame(columns=property_df.columns)
                    st.info("ℹ️ No unmatched records to map")

                # Step 5: Combine matched and unmatched
                st.write("**Step 5:** Combining records...")

                if len(filtered_matched_df) > 0 and len(mapped_unmatched_df) > 0:
                    combined_df = pd.concat([filtered_matched_df, mapped_unmatched_df], ignore_index=True)
                elif len(filtered_matched_df) > 0:
                    combined_df = filtered_matched_df.copy()
                else:
                    combined_df = mapped_unmatched_df.copy()

                st.success(f"✅ Combined total: {len(combined_df):,} records")

                # Step 6: Delete columns
                st.write("**Step 6:** Deleting columns...")
                existing_cols_to_delete = [col for col in COLUMNS_TO_DELETE if col in combined_df.columns]
                combined_df = combined_df.drop(columns=existing_cols_to_delete)
                st.success(f"✅ Deleted {len(existing_cols_to_delete)} columns")

                # Step 7: Add new columns
                st.write("**Step 7:** Adding Mail_CallRail, Lead_Type, Mail_Type...")

                # Rotate through codes
                mail_callrail_values = [codes[i % len(codes)] for i in range(len(combined_df))]
                combined_df['Mail_CallRail'] = mail_callrail_values
                combined_df['Lead_Type'] = lead_type
                combined_df['Mail_Type'] = mail_type

                st.success("✅ Added 3 new columns")

                # Step 8: Deduplicate on AGGR_GROUP
                st.write("**Step 8:** Deduplicating on AGGR_GROUP...")
                initial_count = len(combined_df)

                if 'AGGR_GROUP' in combined_df.columns:
                    # Only dedupe rows that have an AGGR_GROUP value
                    has_aggr = combined_df['AGGR_GROUP'].notna() & (combined_df['AGGR_GROUP'] != '')

                    df_with_aggr = combined_df[has_aggr].drop_duplicates(subset=['AGGR_GROUP'])
                    df_without_aggr = combined_df[~has_aggr]

                    combined_df = pd.concat([df_with_aggr, df_without_aggr], ignore_index=True)

                    removed_dupes = initial_count - len(combined_df)
                    st.success(f"✅ Removed {removed_dupes:,} duplicates")
                else:
                    st.info("ℹ️ AGGR_GROUP column not found, skipping deduplication")

                # Step 9: Apply title case
                st.write("**Step 9:** Applying title case...")
                combined_df = apply_title_case(combined_df)
                st.success("✅ Applied title case (ALL CAPS for state columns)")

                # Step 10: Rename columns
                st.write("**Step 10:** Renaming columns...")
                existing_renames = {old: new for old, new in COLUMN_RENAMES.items() if old in combined_df.columns}
                combined_df = combined_df.rename(columns=existing_renames)
                st.success(f"✅ Renamed {len(existing_renames)} columns")

                # Store in session state
                st.session_state['processed_df'] = combined_df
                st.session_state['processing_complete'] = True

                # Summary
                st.subheader("📊 Processing Summary")
                st.write(f"• **Original landlocked records:** {len(landlocked_df):,}")
                st.write(f"• **Matched to Property Search:** {len(matched_df):,}")
                st.write(f"• **Excluded (owner has nearby property):** {excluded_count:,}")
                st.write(f"• **Unmatched (added from landlocked file):** {len(unmatched_landlocked_df):,}")
                st.write(f"• **Final record count:** {len(combined_df):,}")

        except Exception as e:
            st.error(f"❌ Error processing files: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

# Download Section
if st.session_state.get('processing_complete', False):
    st.subheader("📥 Download Processed File")

    processed_df = st.session_state['processed_df']

    # Preview
    st.write("**Preview (first 10 rows):**")
    st.dataframe(processed_df.head(10), use_container_width=True)

    # File naming
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_name = f"landlocked_filtered_{current_time}.csv"
    filename = st.text_input("Output filename:", value=default_name)
    if not filename.endswith('.csv'):
        filename += '.csv'

    # Download button
    csv_data = processed_df.to_csv(index=False)

    st.download_button(
        label=f"📄 Download {filename}",
        data=csv_data,
        file_name=filename,
        mime="text/csv"
    )

    # Column list
    with st.expander("📋 Final Column List"):
        for i, col in enumerate(processed_df.columns, 1):
            st.write(f"{i}. {col}")

else:
    st.info("👆 Upload files and configure settings to start processing")
