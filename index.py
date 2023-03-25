'''test Flask'''

from flask import Flask, render_template, redirect, url_for
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired

from app import translate, generate

app = Flask(__name__)

app.config['SECRET_KEY'] = 'completely_unique_secret_keyZfAb'

Bootstrap(app)

class TranslateForm(FlaskForm):
    name = StringField('Insert your Shakespearean English here', validators=[DataRequired()])
    submit = SubmitField('Translate')

class PromptForm(FlaskForm):
    name = StringField('Insert your Shakespearean prompt here', validators=[DataRequired()])
    submit = SubmitField('Generate text')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/translator', methods=['GET', 'POST'])
def translator():
    translate_form = TranslateForm()

    original = ''
    translation = ''

    if translate_form.validate_on_submit():
        original = translate_form.name.data
        translation = translate(original)

    return render_template(
        'translator.html',
        form=translate_form,
        original=original,
        translation=translation,
        )

@app.route('/generator', methods=['GET', 'POST'])
def generator():
    prompt_form = PromptForm()

    prompt = ''
    generation = ''

    if prompt_form.validate_on_submit():
        prompt = prompt_form.name.data
        generation = generate(prompt)

    return render_template(
        'generator.html',
        form=prompt_form,
        prompt=prompt,
        generation=generation
        )

