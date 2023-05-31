import numpy as np
import pandas as pd
from unidecode import unidecode
from datetime import datetime
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def cleaning_phone(df, column):

    df_copy = df.copy()

    df_copy.loc[:, 'Telemovel_clean'] = df_copy[column].str.replace(r'[^0-9+]', '')

    # delete spaces in the phone numbers
    df_copy.loc[:,'Telemovel_clean'] = df_copy['Telemovel_clean'].str.replace(" ", "")

    # replace +351 by blank
    df_copy.loc[:, 'Telemovel_clean'] = df_copy['Telemovel_clean'].replace(r'^\+351', '', regex=True)

    # create a new column with the length of each telemovel, valid telemovel has a length of 9
    df_copy.loc[:, 'Telemovel_Count'] = df_copy['Telemovel_clean'].str.len()

    # Filter by all the telemovels that are not blank
    df_copy = df_copy[df_copy['Telemovel_clean'] != '']

    # Keep all the telemovels that star by 9
    df_copy = df_copy.loc[df_copy['Telemovel_clean'].str.startswith('9', na=False)]

    # Keep all the clients with phone number with length equal to 9
    df_copy = df_copy[df_copy['Telemovel_Count'] == 9]

    return df_copy

def create_birthday(df, date_column):

    today =pd.to_datetime('today', format='%Y-%m-%d')
    # initialize an empty list to store out-of-bounds dates
    out_of_bounds_dates = []

    # loop over the series and check for out-of-bounds dates
    for date in df[date_column]:
        try:
            pd.to_datetime(date)
        except pd.errors.OutOfBoundsDatetime:
            out_of_bounds_dates.append(date)

    for i, date in zip(range(len(out_of_bounds_dates)), out_of_bounds_dates):
        new_date = out_of_bounds_dates[i].replace(year=int('19'+str(out_of_bounds_dates[i].year)[-2:]))
        df.loc[df[date_column] == date, date_column] = new_date

    df.loc[:, date_column] =  pd.to_datetime(df[date_column], format='%Y-%m-%d')
    df['Idade'] = round((today - df[date_column])/ np.timedelta64(1, 'Y'))
    df['Idade'].fillna(-1, inplace=True)
    df['Idade'] = df['Idade'].astype(int)
    
    # create range ages 
    conditions = [
       ((df['Idade'] > 0) & (df['Idade'] <=34)),
        ((df['Idade'] > 34) & (df['Idade'] <=54)),
        ((df['Idade'] > 55)),
        
        ]
    choices = ['[0-34]', '[35-54]', '[55>]']
    df['range_age'] = np.select(conditions, choices, default='No age')

    return df

def create_gender(df, df_female_names, df_male_names):
    # Select a column with firts name
    df[['first_name1', 'second_name', 'last_name']] = df['Nome'].str.split(n=2, expand=True)
    
    df['first_name'] = np.where((df["first_name1"].str.len() <= 2) | (df["first_name1"].isin(['Dra', 'dra']) ), df['second_name'], df['first_name1'] )
    df['first_name'] = df['first_name'].astype(str)
    # change all the names by lower letter
    df['first_name'] = df['first_name'].str.lower()

    # remove accents such as ´
    df['first_name'] = df['first_name'].apply(lambda x: unidecode(x))
    #  Lower letter for the list of names
    df_female_names['Name'] = df_female_names['Name'].str.lower()
    df_male_names['Name'] = df_male_names['Name'].str.lower()

    # Create two lists, one with the male names and other with the female names

    woman_names = list(df_female_names['Name'].unique())
    man_names = list(df_male_names['Name'].unique())

    # Remove accents from the list of names
    woman_names = [unidecode(name) for name in woman_names]
    man_names = [unidecode(name) for name in man_names]

    # Create column with the gender, if the name is not in a name list the gender will be 'other'
    df['Gender'] = np.where(df['first_name'].isin(woman_names), 'Female',
                                    np.where(df['first_name'].isin(man_names), 'Male', 'Other') )

    # Create a list with the names that has a gender other
    other_names = df[df['Gender']=='Other']['first_name'].unique()

    # List with the male names in the list above
    add_man_names = ['zahir', 'amandio', 'auzier', 'craig', 'jiayn', 'richardt',  'clalton', 'norbeto', 'eugenio', 'rugiatu']

    # list with the name in other_names that are not male
    add_woman_names = []

    # iterate over names in the 'name' column
    for name in other_names:
        if name not in add_man_names:
            add_woman_names.append(name)


    # Extend the list with the precious names
    woman_names.extend(add_woman_names)
    man_names.extend(add_man_names)

    # names that are in the clients dataframe but not in the names lists

    woman_names.extend(['rabia', 'claudina','miquelina', 'lynou',  'aleida', 'ma', 'ernestina', 'rosalinda', 'joanne', 'leopoldina', 'ivete', 'mabilde', 'flamina', 'aurelio', 
                        'creuza', 'flaviana', 'osvar', 'wael', 'nieves', 'djamila', 'malfada', 'arlindo', 'viorica', 'sherline', 'monira'])

    man_names.extend(['mohomed', 'aurelio'])

    # Create gender with the update names lists
    df['Gender'] = np.where(df['first_name'].isin(woman_names), 'Female',
                                    np.where(df['first_name'].isin(man_names), 'Male', 'Other') )
    
    df["First name"] = df["first_name"].str.capitalize()
    df["First name"] = np.where(df["First name"] == 'Nan', '', df["First name"])
    # Delete extra columns
    df.drop(['first_name', 'second_name', 'first_name1', 'last_name'], axis =1, inplace=True)

    return df

def create_time_features(df, date_column):
    df_ = df.copy()
    # Create features month and month name
    df_['Month'] = df_[date_column].dt.month
    df_['Month_name'] = df_[date_column].dt.month_name()
    df_ = df_.sort_values('Month')

    # create feature season
    bins = [0, 3, 6, 9, 12]
    labels = ['Winter', 'Spring', 'Summer', 'Fall']
    df_['season'] = pd.cut(df_[date_column].dt.month, bins=bins, labels=labels)

    return df_

def create_rfm_columns(df):
    now = datetime.now()
    rfm = df.groupby('ID_Cli').agg({'Data' : lambda day : (now - day.max()).days,
                               'ID_Tra': lambda num : len(num),
                              'ValorTotal': lambda price : price.sum()                           
                             })

    col_list = ['Recency','Frequency','Monetary']
    rfm.columns = col_list
    # Recency in months
    rfm['Recency_months'] = rfm['Recency']/30

    rfm.drop('Recency', axis=1, inplace=True)

    return rfm



def percentage_cum_df(df, segment_col, segment_val):
    df = df.reset_index().groupby([segment_col]).agg({segment_val: [ 'mean', 'max', 'min'], 'ID_Cli':'count'}).reset_index()
    multi_index = df.columns
    column_names = [('_'.join(col)).strip('_') for col in multi_index]
    df.columns = column_names
    df['%'] = round((df['ID_Cli_count']/sum(df['ID_Cli_count']))*100, 2)
    df['cumsum_%'] = round(df.loc[::-1, '%'].cumsum()[::-1], 2)
    df['%'] = df['%'].astype(str)
    df['cumsum_%'] = df['cumsum_%'].astype(str)
    df['cumulative_%'] = np.where(df[segment_col] != 1, 'Top'+' '+df['cumsum_%'], 'Bottom'+' '+df['%'])
    df.drop(['cumsum_%'], axis=1, inplace=True)
    
    return df

def produto_cabelo(df):

    brush_cond = df['Obs'].str.contains('br|brush|rush|alisame', case=False)
    corte_cond = df['Obs'].str.contains('ct|corte|cor', case=False)
    color_cond = df['Obs'].str.contains('tinta|colo|madei|luz|henna|descol|aplic', case=False)
    lava_cond = df['Obs'].str.contains('lava|banho|espuma|creme|laca', case=False)
    tratmentos_cond = df['Obs'].str.contains('tratam|idra|cer', case=False)
    hairstyle_cond = df['Obs'].str.contains('apanh|perman|laca|progre', case=False)

    df.loc[:, 'brush'] = np.where(brush_cond & (df['Categoria']== 'Cabelo'), 'Brush', '')
    df.loc[:,'corte'] = np.where(corte_cond & (df['Categoria']== 'Cabelo'), 'Corte', '')
    df.loc[:,'color'] = np.where(color_cond & (df['Categoria']== 'Cabelo'), 'Color', '')
    df.loc[:,'lavagem'] = np.where(lava_cond & (df['Categoria']== 'Cabelo'), 'Lavagem', '')
    df.loc[:,'Tratamentos'] = np.where(tratmentos_cond & (df['Categoria']== 'Cabelo'), 'Tratamento', '')
    df.loc[:,'Hairstyle'] = np.where(hairstyle_cond & (df['Categoria']== 'Cabelo'), 'Hairstyle', '')

    # df.loc[:,'Produto cabelo']= df[['brush', 'corte', 'color', 'lavagem','Tratamentos','Hairstyle']].apply(lambda row: ' '.join([value for value in row if value != '']), axis=1)
    df['Produto cabelo'] = df[['brush', 'corte', 'color', 'lavagem','Tratamentos','Hairstyle']].apply(lambda row: [row[column] for column in ['brush', 'corte', 'color', 'lavagem','Tratamentos','Hairstyle'] if row[column] != ''], axis=1)
        
    df = df.explode('Produto cabelo')
    df.sort_values('ID_Tra', inplace = True)
    # create new id for treatments
    df.loc[:, 'ID_Tra'] = pd.factorize(df.apply(tuple, axis=1))[0] + 1

    df.loc[:, 'Produto'] = np.where(df['Produto cabelo'].isna(),  df['Produto'], df['Produto cabelo'])
    df.drop(['brush', 'corte', 'color', 'lavagem','Tratamentos','Hairstyle', 'Produto cabelo'], axis = 1, inplace = True)
    
    return df

def plot_trents(df, list_features_group, var_count, var_sum):

    if len(list_features_group) == 2:
        fig = px.bar(df.groupby(list_features_group)[var_count].count().reset_index(), x=list_features_group[0], y=var_count, color= list_features_group[1])
        fig.update_layout(barmode='stack', xaxis={'categoryorder': 'total descending'},
                        title_text=f"Number of treatments by {list_features_group[0]} and {list_features_group[1]}",
                        template="plotly_dark",
                        title_font=dict(size=25, color='#8a8d93', family="Lato, sans-serif"),
                        font=dict(color='#8a8d93'),
                        title_x=0.5,
                        hoverlabel=dict(bgcolor="#f2f2f2", font_size=13, font_family="Lato, sans-serif")
                        )
        fig.show(width=800, height=500)

        # Employees by ValorTotal an categoria
        fig = px.bar(df.groupby(list_features_group)[var_sum].sum().reset_index(), x=list_features_group[0], y=var_sum, color= list_features_group[1])
        fig.update_layout(barmode='stack', xaxis={'categoryorder': 'total descending'},
                        title_text=f"Valor total for {list_features_group[0]} and {list_features_group[1]}",
                        template="plotly_dark",
                        title_font=dict(size=25, color='#8a8d93', family="Lato, sans-serif"),
                        font=dict(color='#8a8d93'),
                        title_x=0.5,
                        hoverlabel=dict(bgcolor="#f2f2f2", font_size=13, font_family="Lato, sans-serif")
                        )

        fig.show(width=800, height=500)
    if len(list_features_group) == 3:
        nun = df[list_features_group[0]].unique()
        num_facet_col = np.where(len(nun) <= 3, 0, np.where((len(nun) <=5) & (len(nun) >3), 2, 3))
        n = np.int16(num_facet_col).item()
        df_plot = df.groupby(list_features_group)[var_count].count().reset_index()
        df_plot.sort_values([ list_features_group[0], var_count], ascending=[True,False], inplace=True)
        fig = px.bar(df_plot, x=list_features_group[1], y=var_count, color=list_features_group[2], 
                    facet_col=list_features_group[0], facet_col_wrap=n)
        fig.update_layout(barmode='stack',
                        title_text=f"Most common treatments by {list_features_group[0]}",
                        template="plotly_dark",
                        title_font=dict(size=25, color='#8a8d93', family="Lato, sans-serif"),
                        font=dict(color='#8a8d93'),
                        title_x=0.5,
                        hoverlabel=dict(bgcolor="#f2f2f2", font_size=13, font_family="Lato, sans-serif")
                        )
        fig.show(width=800, height=500)


        df_plot = df.groupby(list_features_group)[var_sum].sum().reset_index()
        df_plot.sort_values([ list_features_group[0], var_sum], ascending=[True,False], inplace=True)
        
        fig = px.bar(df_plot, x=list_features_group[1], y=var_sum, color=list_features_group[2], 
                    facet_col=list_features_group[0], facet_col_wrap=n)
        fig.update_layout(barmode='stack',
                        title_text=f"Valor Total for treatments by {list_features_group[0]}",
                        template="plotly_dark",
                        title_font=dict(size=25, color='#8a8d93', family="Lato, sans-serif"),
                        font=dict(color='#8a8d93'),
                        title_x=0.5,
                        hoverlabel=dict(bgcolor="#f2f2f2", font_size=13, font_family="Lato, sans-serif")
                        )
        fig.show(width=800, height=500)

def filter_Category_month(df, gender, category, months_list, contain_category = True):
    '''
    df: dataframe
    gender: female, Male, other
    category: Desired category
    month_list: list of months to analyze
    contain_category: if it's True, it shows all the clients that did treatment in the category x
    ''' 
    
    df_filter_gender = df[(df['Gender'] == gender) & (df['Month_name'].isin(months_list))]
    df_filter_gender = df_filter_gender.groupby(['ID_Cli', 'Categoria', 'Produto']).agg(count=('ID_Cli', 'count')).reset_index()
    
    if category == 'NO':
        
        
        df_empregado = df_filter_gender.merge(df[['Empregados', 'ID_Cli']], on='ID_Cli', how='left')
        df_empregado = df_empregado.groupby(['ID_Cli', 'Categoria', 'Produto', 'Empregados']).agg(count=('Empregados', 'count')).reset_index()

        df_group_plot = df_filter_gender.groupby(['Categoria', 'Produto'])['count'].sum().reset_index()
        fig = px.bar(df_group_plot, x='Categoria', y='count', color= 'Produto')
        fig.update_layout(barmode='stack', xaxis={'categoryorder': 'total descending'}, template="plotly_dark")
        fig.show()
    
    else:
        if contain_category == True:
            
            df_category = df_filter_gender[df_filter_gender['Categoria'] == category]
            list_ids = list(df_category['ID_Cli'].unique())
            
            df_filter_gender = df_filter_gender[df_filter_gender['ID_Cli'].isin(list_ids)]
            df_filter_gender.fillna(0, inplace=True)

            df_empregado = df_filter_gender.merge(df[['Empregados', 'ID_Cli']], on='ID_Cli', how='left')
            df_empregado = df_empregado.groupby(['ID_Cli', 'Categoria', 'Produto', 'Empregados']).agg(count=('Empregados', 'count')).reset_index()

            df_group_plot = df_filter_gender.groupby(['Categoria', 'Produto'])['count'].sum().reset_index()
            fig = px.bar(df_group_plot, x='Categoria', y='count', color= 'Produto')
            fig.update_layout(barmode='stack', xaxis={'categoryorder': 'total descending'}, template="plotly_dark")
            fig.show(width=800, height=500)
        else:

            df_category = df_filter_gender[df_filter_gender['Categoria'] == category]
            list_ids = list(df_category['ID_Cli'].unique())
            
            df_filter_gender = df_filter_gender[~df_filter_gender['ID_Cli'].isin(list_ids)]
            df_filter_gender.fillna(0, inplace=True)

            df_empregado = df_filter_gender.merge(df[['Empregados', 'ID_Cli']], on='ID_Cli', how='left')
            df_empregado = df_empregado.groupby(['ID_Cli', 'Categoria', 'Produto', 'Empregados']).agg(count=('Empregados', 'count')).reset_index()

            df_group_plot = df_filter_gender.groupby(['Categoria', 'Produto'])['count'].sum().reset_index()
            fig = px.bar(df_group_plot, x='Categoria', y='count', color= 'Produto')
            fig.update_layout(barmode='stack', xaxis={'categoryorder': 'total descending'}, template="plotly_dark")
            fig.show(width=800, height=500)
            
    return df_filter_gender, df_empregado

def segments_categoria_gender(df, seg):

    df_bar = df.groupby('Categoria')['ID_Cli'].nunique().reset_index()
    df_pie = df.groupby('Gender')['ID_Cli'].nunique().reset_index()
    df_bar_M = df[df['Gender'] == 'Male'].groupby(['Categoria'])['ID_Cli'].nunique().reset_index()
    df_bar_F = df[df['Gender'] == 'Female'].groupby(['Categoria'])['ID_Cli'].nunique().reset_index()

    df_bar['color'] ='rgb(102,194,165)'
    df_bar_F['color'] ='rgb(102,194,165)'
    df_bar_M['color'] ='rgb(252,141,98)'


    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "bar"}, {"type": "pie"}],
                [{"colspan": 2}, None]],
        column_widths=[0.8, 0.2], vertical_spacing=0.2, horizontal_spacing=0.02,
        subplot_titles=("Categoria","Gender", "Categoria by gender")
    )


    fig.add_trace(go.Bar(x=df_bar['Categoria'], y=df_bar['ID_Cli'],marker=dict(color=df_bar['color'] ),
                name='Age'), row=1, col=1)



    fig.add_trace(go.Pie(labels=df_pie['Gender'], values=df_pie['ID_Cli'], 
                marker=dict(colors=['rgb(102,194,165)', 'rgb(252,141,98)', 'rgb(141,160,203)']),
                hole=0.5, hoverinfo='label+percent+value', textinfo='label'),
                row=1, col=2)

    fig.add_trace(go.Bar(x=df_bar_F['Categoria'], y=df_bar_F['ID_Cli'], 
                        marker=dict(color= df_bar_F['color']), name='Female'),
                        row=2, col=1)
    fig.add_trace(go.Bar(x=df_bar_M['Categoria'], y=df_bar_M['ID_Cli'], 
                        marker=dict(color= df_bar_M['color']), name='Male'),
                        row=2, col=1)


    fig.update_yaxes(showgrid=False, ticksuffix=' ',  row=1, col=1)
    fig.update_xaxes(tickmode = 'array',  row=1, col=1, categoryorder='total descending')
    fig.update_xaxes(tickmode = 'array', row=2, col=1, categoryorder='total descending')
    fig.update_yaxes(visible=False, row=2, col=1)
    fig.update_layout(barmode='stack', height=600, bargap=0.2,
                    margin=dict(b=0.04,r=20,l=20), xaxis=dict(tickmode='linear'),
                    title_text=f"Analyzing Clients for the segement {seg}",
                    template="plotly_dark",
                    title_font=dict(size=25, color='#8a8d93', family="Lato, sans-serif"),
                    font=dict(color='#8a8d93'),
                    title_x=0.5,
                    hoverlabel=dict(bgcolor="#f2f2f2", font_size=13, font_family="Lato, sans-serif"),
                    showlegend=False)
    fig.show(width=800, height=500)


