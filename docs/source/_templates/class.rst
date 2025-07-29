{{ name | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :show-inheritance:
   :inherited-members:
   :member-order: groupwise
   :exclude-members: {{ attributes|join(', ') }}

   {% if attributes %}
   .. rubric:: Attributes and Properties

   .. autosummary::
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {% endfor %}

   {% endif %}

   {% if methods %}
   .. rubric:: Methods

   {% endif %}